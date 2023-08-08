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
from torch import nn, Tensor
from torchmetrics import BLEUScore
from transformers import AutoTokenizer
from modules.MHBAMixer import MHBAMixer


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x + self.pe[: x.size(0), :, :]  # type: ignore[index]
        return self.dropout(x)


class EmbeddingWithPosition(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_len, max_batch_size):
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


class BackBoneModule(nn.Module):
    def __init__(self, vocab_size, n_heads, embedding_dim, hidden_dim, kernel_size, padding, prob, num_mixers):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        # n_heads, hidden_dim: int, kernel_size: int, padding: int, prob: float, num_mixers
        self.backbone = MHBAMixer(n_heads, embedding_dim, hidden_dim, kernel_size, padding, prob,  num_mixers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, vocab_size),
            nn.LayerNorm(vocab_size),
            nn.Softmax(dim=-1)
        )
        self.criterion = nn.CrossEntropyLoss()

    @staticmethod
    def greek_decoder(token_logit):
        token_idx = torch.argmax(token_logit, dim=-1)
        return token_idx

    def forward(self, inputs, target, padding_mask, causal_mask, embedding):
        feature_embedding = embedding(inputs)
        last_token_feature = self.backbone(feature_embedding, padding_mask, causal_mask)[:, -1]
        last_token_logit = self.classifier(last_token_feature)
        # print(last_token_logit.dtype, )
        most_prob_idx = self.greek_decoder(last_token_logit)
        loss = self.criterion(last_token_logit, target.long())
        return last_token_logit, most_prob_idx, loss


class GenerateModule(LightningModule):
    def __init__(self,
                 vocab_size: int=21128,
                 embedding_dim: int=64,
                 n_heads: int=4,
                 max_len: int=64,
                 hidden_dim: int=128,
                 kernel_size = [ 5, 3, 3, 3, 3, 3, 3, 7 ],
                 padding = [2, 1, 1, 1, 1, 1, 1, 3],
                 lr: float=0.001,
                 b1: float=0.999,
                 b2: float=0.9,
                 num_layers: int=2,
                 prob: float=0.8,
                 batch_size: int=64
                 ):
        super().__init__()
        # define tokenizer convert ids to words
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
        self.embedding = EmbeddingWithPosition(vocab_size, embedding_dim, max_len, batch_size)
        # init generator module
        self.model = BackBoneModule(vocab_size, n_heads, embedding_dim, hidden_dim, kernel_size, padding, prob, num_layers)
        # init discriminator
        self.classification_loss = nn.CrossEntropyLoss()
        self.save_hyperparameters()
        # define temporary variable
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outpus = []
        self.automatic_optimization = False
        self.blue = BLEUScore(smooth=True)
        self.causal_mask = torch.tril(torch.ones(max_len, max_len), diagonal=0).bool().cuda()

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def compute_metrics(metrics):
        batch_len = len(metrics)
        average_loss = 0.
        average_blue = 0.
        for iter in metrics:
            average_loss += iter[0]
            average_blue += iter[1]
        return average_loss/batch_len, average_blue/batch_len

    def covert_tokens_to_sentences(self, tokens):
        sentences = self.tokenizer.batch_decode(tokens.long(), skip_special_tokens=True)
        return sentences

    def share_step(self, batch):
        src, target, token_type_ids, attention_mask = batch
        memory = torch.empty(target.size(0), 0).long().cuda()
        total_loss = 0.
        for len in range(self.hparams.max_len):
            output, idx, loss = self.model(
                src, target[:, len], attention_mask, self.causal_mask[len][: src.size(1)], self.embedding)
            memory = torch.cat((memory, idx[:, None]), dim=1)
            total_loss += loss
        return memory, total_loss

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        optimizer = self.optimizers()
        sch = self.lr_schedulers()
        self.untoggle_optimizer(optimizer)
        memory, loss = self.share_step(batch)
        loss = loss/(batch[0].size(0)*batch[0].size(1))
        self.manual_backward(loss)
        optimizer.step()
        sch.step()
        pred = self.covert_tokens_to_sentences(memory)
        del memory
        ground_truth = self.covert_tokens_to_sentences(batch[1])
        blue_score = self.blue(pred, ground_truth)
        del pred, ground_truth
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_blue", blue_score)
        self.training_step_outputs.append((loss, blue_score))
        return loss

    def on_train_epoch_end(self) -> None:
        average_loss, average_blue = self.compute_metrics(self.training_step_outputs)
        self.log("train_loss_epoch", average_loss, prog_bar=True, on_epoch=True)
        self.log("train_blue_epoch", average_blue, prog_bar=True, on_epoch=True)
        self.training_step_outputs = []

    def validation_step(self, batch, batch_idx):
        memory, loss = self.share_step(batch)
        loss = loss / (batch[0].size(0) * batch[0].size(1))
        pred = self.covert_tokens_to_sentences(memory)
        del memory
        ground_truth = self.covert_tokens_to_sentences(batch[1])
        blue_score = self.blue(pred, ground_truth)
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("val_blue", blue_score)
        print(f"original sentence:\t{ground_truth[0]}")
        print(f"generate sentence:\t{pred[0]}")
        del pred, ground_truth
        self.validation_step_outputs.append((loss, blue_score))

    def on_validation_epoch_end(self) -> None:
        average_loss, average_blue = self.compute_metrics(self.validation_step_outputs)
        self.log("val_loss_epoch", average_loss, prog_bar=True, on_epoch=True)
        self.log("val_blue_epoch", average_blue, prog_bar=True, on_epoch=True)
        self.validation_step_outputs = []

    def test_step(self, batch, batch_idx):
        memory, loss = self.share_step(batch)
        loss = loss / (batch.size(0) * batch.size(1))
        pred = self.covert_tokens_to_sentences(memory)
        del memory
        ground_truth = self.covert_tokens_to_sentences(batch[1])
        blue_score = self.blue(pred, ground_truth)
        self.log("test_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("test_blue", blue_score)
        print(f"original sentence:\t{ground_truth[0]}")
        print(f"generate sentence:\t{pred[0]}")
        del pred, ground_truth
        self.test_step_outpus.append((loss, blue_score))

    def on_test_epoch_end(self) -> None:
        average_loss, average_blue = self.compute_metrics(self.test_step_outputs)
        self.log("test_loss_epoch", average_loss, prog_bar=True, on_epoch=True)
        self.log("test_blue_epoch", average_blue, prog_bar=True, on_epoch=True)
        self.test_step_outputs = []


    def configure_optimizers(self):
        opt = torch.optim.SGD(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2)
        return [opt], [lr_scheduler]