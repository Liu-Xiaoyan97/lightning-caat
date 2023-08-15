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
        self.activate = nn.GELU()

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
        self.activate = nn.GELU()
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
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(intermediate_dim, hidden_dim),
        ])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)