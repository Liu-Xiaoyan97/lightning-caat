#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/8/7 16:22
# @Author  : lxy15058247683@aliyun.com
# @FileName: MHBAMixer.py
# @Copyright: MIT
# 作者 ：D_wh
# 时间 ：2023/7/16 16:43
# 格式化 ：Ctrl+Alt+L
# 清除不用代码 ：Ctrl+Alt+O
# 智能导入 ：Alt+Enter
from math import sqrt
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops.layers.torch import Rearrange


class Bernolli_sampling_nlp:
    def __init__(self, max_len=100, prob=1):
        self.max_len = max_len
        self.prob = prob

    def __call__(self, inputs):
        n_samples, max_seq_len, embedding_dim = inputs.size(0), inputs.size(1), inputs.size(-1)
        Benoulli = torch.distributions.bernoulli.Bernoulli(self.prob)
        masks = Benoulli.sample((n_samples, max_seq_len)).unsqueeze(-1).repeat(1, 1, embedding_dim).bool().cuda()
        inputs = F.softmax(inputs.masked_fill(~masks, -np.inf), dim=-2)
        return inputs


class Bernolli_sampling_cv:
    def __init__(self, max_len=100, prob=1):
        self.max_len = max_len
        self.prob = prob

    def __call__(self, inputs):
        n_samples, max_seq_len, embedding_dim = inputs.size(0), inputs.size(1), inputs.size(-1)
        Benoulli = torch.distributions.bernoulli.Bernoulli(self.prob)
        masks = Benoulli.sample((n_samples, max_seq_len, embedding_dim)).cuda().bool()
        inputs = F.softmax(inputs.masked_fill(~masks, -np.inf), dim=-2)
        return inputs


class HBA(nn.Module):
    def __init__(self, mode, max_seq_len, embedding_dim, prob, kernel_size, dilation, padding):
        super(HBA, self).__init__()
        self.embedding_dim = embedding_dim
        self.local_information = nn.Conv1d(embedding_dim, embedding_dim,
                                                          kernel_size, 1, padding, dilation, groups=embedding_dim)
        self.activate = nn.GELU()
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.global_information = nn.Linear(max_seq_len, max_seq_len)
        self.bernolli_sampling = self.Choice_Bernolli(mode)(max_len=max_seq_len, prob=prob)
        self.softmax = nn.Softmax(-1)

    def Choice_Bernolli(self, mode: str):
        if mode == "cv":
            return Bernolli_sampling_cv
        else:
            return Bernolli_sampling_nlp

    def forward(self, x):
        x = x.transpose(1, 2)
        # [N, embedding_dim 4, max_seq_len 384]
        q = self.bn(self.activate(self.local_information(x)+x))
        k = self.activate(self.bernolli_sampling(x))
        v = self.activate(self.global_information(x))
        attention = self.softmax(torch.bmm(q, k.transpose(1, 2))/sqrt(self.embedding_dim))
        output = torch.bmm(attention, v)
        return output.transpose(1, 2), attention


class MHBA(nn.Module):
    """
            :param n_head: 头数
            :param mode: nlp or cv
            :param max_seq_len: 最大序列长度
            :param embedding_dim: 嵌入维度
            :param prob: 采样概率 一般为0.8，消融实验做过了 0.8 最好
            :param kernel_size: 卷积核大小
            :param dilation: 空洞率 0表示普通卷积，以k=3,d=1的卷积为例，近似等于k=5,d=0的卷积
            :param padding: 填充大小，用于处理边界
            .. math:: output_feature_map = lower(\\frac{(l+2p-k)}{s})
            """
    def __init__(self, n_head, mode, max_seq_len, embedding_dim, prob, kernel_size, dilation, padding):
        super(MHBA, self).__init__()
        # print(max_seq_len, n_head)
        assert max_seq_len % n_head == 0, 'max_seq_len must be divisible by the n_head.'
        self.embedding_dim = embedding_dim
        self.n_head = n_head
        self.max_seq_len = max_seq_len
        self.input_dim = int(max_seq_len // n_head)
        self.local_information = nn.Conv1d(embedding_dim, embedding_dim,
                                                          kernel_size, 1, padding, groups=self.input_dim)
        self.activate = nn.GELU()
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.global_information = nn.Linear(self.input_dim, self.input_dim)
        self.mode = mode
        if mode == "cv":
            self.bernolli_sampling = Bernolli_sampling_cv(prob=prob)
        else:
            self.bernolli_sampling = Bernolli_sampling_nlp(prob=prob)
        self.softmax = nn.Softmax(-1)
        self.trans = Rearrange("b (m h) d -> (b h) d m ", h=n_head)
        self.trans2 = Rearrange("(b h) d m  -> b (m h) d ", h=n_head)

    def forward(self, inputs):
        #  b (chw) (p1 p2)
        # print(inputs.shape)
        if self.mode == "cv":
            q = self.trans(inputs)
            k = self.trans(inputs)
            v = self.trans(inputs)
        else:
            q = inputs.view(-1, self.embedding_dim, self.input_dim)
            k = inputs.view(-1, self.embedding_dim, self.input_dim)
            v = inputs.view(-1, self.embedding_dim, self.input_dim)
        # print(q.shape)
        q = self.bn(self.activate(self.local_information(q)+q))
        k = self.activate(self.bernolli_sampling(k))
        v = self.activate(self.global_information(v))
        attention = self.softmax(torch.bmm(q, k.transpose(1, 2)) / sqrt(self.embedding_dim))
        output = torch.bmm(attention, v)
        if self.mode == "cv":
            return self.trans2(output), attention
        else:
            return output.reshape(-1, self.max_seq_len, self.embedding_dim), attention


class MHBAWithMask(nn.Module):
    def __init__(self, num_heads, embed_size, hidden_dim, prob, kernel_size, padding):
        super(MHBAWithMask, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.prob = prob
        assert (
                self.head_dim * num_heads == embed_size
        ), "Embedding size needs to be divisible by num_heads"

        self.values = torch.nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Conv2d(self.num_heads, self.num_heads, kernel_size, 1, padding, groups=self.num_heads)
        self.groupOperation = Rearrange("b l (h d) -> b h l d", h=self.num_heads, d=self.head_dim)
        self.restoreOperation = Rearrange("b h l d -> b l (h d)", h=self.num_heads)
        self.norm1 = nn.BatchNorm2d(self.num_heads)
        self.linear = nn.Linear(self.head_dim, self.head_dim)
        self.norm2 = nn.LayerNorm(self.head_dim)
        self.drop = nn.Dropout(0.1)
        self.activate = nn.ReLU()

    @staticmethod
    def keys(inputs, prob):
        batch, length, size = inputs.size()
        ber_mask = inputs.new_empty(batch, length)
        ber_mask = ber_mask.bernoulli_(prob)
        outputs = inputs.masked_fill(~ber_mask.unsqueeze(-1).bool(), float("-1e20"))
        return F.softmax(outputs, dim=-2)

    def forward(self, query, keys, values, padding_mask, causal_mask):
        mask = padding_mask & causal_mask
        query = self.activate(self.norm1(self.queries(self.groupOperation(query))+self.groupOperation(query)))
        keys = self.activate(self.groupOperation(self.keys(keys, self.prob)))
        values = self.values(self.groupOperation(values))
        energy = self.activate(torch.einsum("nhqd, nhkd->nhqk", [query, keys]))
        energy = energy.masked_fill(~mask.unsqueeze(1).bool(), float("-1e20"))
        attention = F.softmax(energy / (self.embed_size ** (1 / 2)), dim=-1)
        out = self.linear(self.drop(self.norm2(torch.einsum("nhql,nhld->nhqd", [attention, values]))))
        out = self.restoreOperation(out)
        return out


class MHBAMixer(nn.Module):
    """n_heads, hidden_dim: int, kernel_size: int, padding: int, prob: float, num_mixers"""
    def __init__(self, n_heads, embedding_dim, hidden_dim: int, kernel_size: int, padding: int, prob: float, num_mixers, **kwargs):
        super(MHBAMixer, self).__init__()
        self.mixers = nn.ModuleList(
            [MixerLayer(n_heads, embedding_dim, hidden_dim, kernel_size[i], padding[i], prob, **kwargs)
                                  for i in range(num_mixers)]
        )

    def forward(self, inputs, padding_mask, causal_mask) -> torch.Tensor:
        for block in self.mixers:
            inputs = block(inputs, padding_mask, causal_mask)
        return inputs


class MixerLayer(nn.Module):
    def __init__(self, n_heads, embedding_dim, hidden_dim: int, kernel_size: int, padding: int, prob: float, **kwargs):
        super(MixerLayer, self).__init__(**kwargs)
        self.kernel_size, self.padding = kernel_size, padding
        self.mwm = MHBAWithMask(n_heads, embedding_dim, hidden_dim, prob, [1, kernel_size], [0, padding])
        self.activate = nn.Tanh()
        self.dropout = nn.Dropout(p=0.5)
        self.ffn = MlpLayer(embedding_dim, hidden_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, inputs, padding_mask, causal_mask) -> torch.Tensor:
        residual = inputs
        outputs = self.mwm(inputs, inputs, inputs, padding_mask, causal_mask)
        outputs = self.activate(outputs + residual)
        residual = outputs
        outputs = self.norm(self.activate(self.ffn(self.dropout(outputs)) + residual))
        return outputs


class MlpLayer(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int, **kwargs):
        super(MlpLayer, self).__init__(**kwargs)
        self.layers = nn.Sequential(*[
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(intermediate_dim, hidden_dim),
        ])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)