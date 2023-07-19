#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/7/10 17:00
# @Author  : lxy15058247683@aliyun.com
# @FileName: lightningmodules.py
# @Copyright: MIT
import json
import math

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning import LightningModule
from tokenizers import tokenizers
from torch import nn, Tensor
import torch.nn.functional as F
from torch.autograd import Variable
from torchmetrics import Accuracy
from transformers import PreTrainedTokenizer, AutoTokenizer


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x + self.pe[: x.size(0), :]  # type: ignore[index]
        return self.dropout(x)

class EmbeddingWithPosition(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_len):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_embedding = PositionalEncoding(embedding_dim, 0.1, max_len)

    def forward(self, tokens):
        feature_embedding = self.word_embedding(tokens)
        feature_embedding = self.positional_embedding(feature_embedding)
        return feature_embedding


def get_key_padding_mask(tokens, pad_id):
    key_padding_mask = torch.zeros(tokens.size(), dtype=torch.float)
    key_padding_mask[tokens.cpu() == pad_id] == (-math.inf)
    return key_padding_mask.cuda()


class Encoder_Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_len, d_model, nhead, dim_feedforward, num_layers, pad_id):
        super().__init__()
        self.pad_id = pad_id
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, batch_first=True, activation=F.gelu), num_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, batch_first=True), num_layers
        )
        self.embedding_module = EmbeddingWithPosition(vocab_size, embedding_dim, max_len)
        self.generate_square_subsequent_mask = nn.Transformer.generate_square_subsequent_mask(max_len).cuda()
        # print(self.generate_square_subsequent_mask)
        self.norm = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.LayerNorm(dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(dim_feedforward, d_model),
            nn.Softmax(dim=-1)
        )

    def forward(self, tensor_src, tensor_tgt):
        # encoder forward
        src_embeddings = self.embedding_module(tensor_src)
        # print(tensor_src, self.pad_id)
        src_key_padding_mask = get_key_padding_mask(tensor_src, self.pad_id)
        src_latent = self.encoder(src_embeddings, src_key_padding_mask=src_key_padding_mask)
        # decoder forward
        tgt_embeddings = self.embedding_module(tensor_tgt)
        tgt_key_padding_mask = get_key_padding_mask(tensor_tgt, self.pad_id)
        tgt_latent = self.decoder(tgt_embeddings, src_latent,
                                            tgt_mask=self.generate_square_subsequent_mask.bool(),
                                            memory_key_padding_mask=src_key_padding_mask,
                                            tgt_key_padding_mask=tgt_key_padding_mask)
        return src_latent, tgt_latent


def covert_distribution_to_max_index(inputs):
    return torch.argmax(inputs, dim=-1)


def compute_accuracy(inputs, target):
    return torch.eq(inputs, target).float().mean()


def merge_metrics(logdict):
    n_steps = len(logdict["gen_loss"])+1e-9
    return sum(logdict["gen_loss"])/n_steps,\
            sum(logdict['dis_loss'])/n_steps

class CAATModule(LightningModule):
    def __init__(self,
                 vocab_size: int=21128,
                 embedding_dim: int=64,
                 d_model: int=64,
                 nhead: int=4,
                 max_len: int=64,
                 dim_feedforward: int=128,
                 lr: float=0.001,
                 b1: float=0.999,
                 b2: float=0.9,
                 num_layers: int=2,
                 num_class: int=2
                 ):
        super().__init__()
        # define tokenizer convert ids to words
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
        # init generator module
        self.generator = Encoder_Decoder(vocab_size, embedding_dim, max_len, d_model,
                                                    nhead, dim_feedforward, num_layers,
                                         self.tokenizer.pad_token_id)
        # init discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, num_class),
            nn.Sigmoid()
        )
        # define loss functions
        self.classification_loss = nn.CrossEntropyLoss()
        self.src_tgt_unsimilarity_loss = nn.CosineSimilarity(dim=-1)
        self.restruct_loss = nn.CrossEntropyLoss(reduction="mean", label_smoothing=0.5)
        # init more lightning hyperparameter
        self.automatic_optimization = False
        self.save_hyperparameters()

        # init generate weight covert latent to vocab probability
        self.generate_linear = nn.Linear(d_model, vocab_size)

        # define temporary variable
        self.training_step_outputs = {"gen_loss": [], "dis_loss": []}
        self.validation_step_outpus = {"gen_loss": [], "dis_loss": []}

    def share_step(self, batch, batch_idx, greek_decode=False):
        tensor_src, tensor_src_mask, tensor_tgt, tensor_tgt_mask, tensor_labels = batch
        tensor_tgt_y = tensor_src
        """
           get src and tgt latent representation. 
           src_latent src_latent [B, L, D] 
           B: batch size, L: hyperparameter max_len -> maximum sequence length, 
           D: hyperparameter d_model ->embedding dimensions
        """
        src_latent, tgt_latent = self.generator(tensor_src, tensor_tgt)
        if greek_decode:
            self.greek_decode(src_latent)
        src_latent_copy, tgt_latent_copy = src_latent.detach(), tgt_latent.detach()
        """
           get src class distribution. 
           cls_src [B, D]
        """
        cls_src = self.discriminator(src_latent_copy.sum(1))
        # cls_src = self.discriminator(src_latent.sum(1))
        """
           get tgt class distribution 
           cls_tgt [B, D]
        """
        cls_tgt = self.discriminator(tgt_latent_copy.sum(1))
        # cls_tgt = self.discriminator(tgt_latent.sum(1))
        """
            reverse sentence label 0->1, 1->0
        """
        reverse_tensor_label = ~tensor_labels.bool()

        """compute classification loss """
        src_cls_loss = self.compute_classification_loss(cls_src, tensor_labels.long())
        tgt_cls_loss = self.compute_classification_loss(cls_tgt, reverse_tensor_label.long())
        discriminator_loss = tgt_cls_loss

        """convert tgt_latent to id probability distribution.
           tgt_latent [B, L, D]
           tgt_word_probability [B, L, W]
           W: hyperparameter vocab_size -> vocabulary size
        """
        tgt_word_probability_distribution = self.covert_latent_to_vocab_probability_distribution(tgt_latent)

        """
            compute restructure loss
        """
        restructure_loss = self.compute_restruct_loss(tgt_word_probability_distribution, tensor_tgt_y, tensor_src_mask)
        """
            compute class index both of src and tgt
            src_class [B, C]
            tgt_class [B, C]
            C: hyperparameter num_classes -> number of classes
        """
        src_class = covert_distribution_to_max_index(cls_src)
        tgt_class = covert_distribution_to_max_index(cls_tgt)

        """
            greek decode
        """
        if greek_decode:
            self.greek_decode(tgt_word_probability_distribution)
        return discriminator_loss, restructure_loss, src_class, tgt_class, tensor_labels, reverse_tensor_label

    def covert_latent_to_vocab_probability_distribution(self, input_tensor):
        vocab_tensor = self.generate_linear(input_tensor)
        log_softmax = F.softmax(vocab_tensor, dim=-1)
        return log_softmax

    def compute_classification_loss(self, logit, tensor_labels):
        return self.classification_loss(logit, tensor_labels.long())

    def compute_triplet_loss(self, positive, negitive, truth):
        return self.triplet_loss(positive, negitive, truth)

    def compute_restruct_loss(self, generate_tensor, tensor_tgt_y, tensor_src_mask):
        return self.restruct_loss(generate_tensor.contiguous().view(-1, generate_tensor.size(-1)),
                                  tensor_tgt_y.long().contiguous().view(-1))/tensor_src_mask.sum()

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        optimizer_generator, optimizer_discriminator = self.optimizers()
        discriminator_loss, restructure_loss, src_class, \
            tgt_class, src_label, tgt_label = self.share_step(batch, batch_idx)

        """ 
            generator backward
        """
        # print(f"batch {batch_idx} before backward-->", src_latent.grad_fn, tgt_latent.grad_fn)
        self.toggle_optimizer(optimizer_generator)
        self.manual_backward(restructure_loss)
        optimizer_generator.step()
        optimizer_generator.zero_grad()
        # for param in self.generator.decoder.parameters():
        #     print(param[0])
        #     break
        self.untoggle_optimizer(optimizer_generator)
        # print(f"batch {batch_idx} after backward-->", src_latent.grad_fn, tgt_latent.grad_fn)
        """
            discriminator backward
        """
        self.toggle_optimizer(optimizer_discriminator)
        self.manual_backward(discriminator_loss)
        optimizer_discriminator.step()
        optimizer_discriminator.zero_grad()
        self.untoggle_optimizer(optimizer_discriminator)
        """
            compute and log metrics records
        """
        self.log("discriminator_loss", discriminator_loss, prog_bar=True)
        self.log("res_loss", restructure_loss, prog_bar=True)
        accuracy = compute_accuracy(src_class, src_label)
        # self.log("src_cls_acc", accuracy, prog_bar=True)
        accuracy = compute_accuracy(tgt_class, tgt_label)
        # self.log("tgt_cls_acc", accuracy, prog_bar=True)
        total_loss = discriminator_loss+restructure_loss
        self.training_step_outputs["gen_loss"].append(restructure_loss)
        self.training_step_outputs["dis_loss"].append(discriminator_loss)
        return total_loss

    def on_train_epoch_end(self) -> None:
        metrics = merge_metrics(self.training_step_outputs)
        self.log("train_gen_loss", metrics[0], prog_bar=True, on_epoch=True)
        self.log("train_dis_loss", metrics[1], prog_bar=True, on_epoch=True)
        self.training_step_outputs["gen_loss"] = []
        self.training_step_outputs["dis_loss"] = []

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=0.001, betas=(self.hparams.b1, self.hparams.b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=0.0001, betas=(self.hparams.b1, self.hparams.b2))
        return [opt_g, opt_d]

    def greek_decode(self, input_tensor):
        output_max_indices = torch.argmax(input_tensor, dim=-1)
        text = self.tokenizer.batch_decode(output_max_indices)
        print("{}".format(text[0]))
        return

    def validation_step(self, batch, batch_idx):
        discriminator_loss, restructure_loss, src_class, \
            tgt_class, src_label, tgt_label = self.share_step(batch, batch_idx, True)
        # print("ground truth -> ", src_label)
        # print("src class    -> ", src_class)
        # print("tgt  truth   -> ", tgt_label.long())
        # print("tgt class    -> ", tgt_class)
        accuracy = compute_accuracy(src_class, src_label)
        # self.log("val_src_cls_acc", accuracy, prog_bar=True)
        accuracy = compute_accuracy(tgt_class, tgt_label)
        # self.log("val_cls_acc", accuracy, prog_bar=True)
        self.validation_step_outpus["gen_loss"].append(restructure_loss)
        self.validation_step_outpus["dis_loss"].append(discriminator_loss)
        # total_loss = src_cls_loss + tgt_cls_loss + restructure_loss

    def on_validation_epoch_end(self) -> None:
        metrics = merge_metrics(self.validation_step_outpus)
        self.log("val_gen_loss", metrics[0], prog_bar=True, on_epoch=True)
        self.log("val_dis_loss", metrics[1], prog_bar=True, on_epoch=True)
        self.validation_step_outpus["gen_loss"] = []
        self.validation_step_outpus["dis_loss"] = []
