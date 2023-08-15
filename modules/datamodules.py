#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/7/10 15:28
# @Author  : lxy15058247683@aliyun.com
# @FileName: datamodules.py
# @Copyright: MIT
import itertools
import json
import os
import re

import numpy as np
import pandas
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader, Dataset, IterableDataset
from datasets import load_dataset
from typing import List, Optional, Dict
from transformers import PreTrainedTokenizer, AutoTokenizer
import os.path
from datasets import load_dataset
from pathlib import Path
import platform
if platform.system() == "Linux":
    num_workers = 40
else:
    num_workers = 1

class DataModules(LightningDataModule):
    def __init__(self, file_path_dir: str, batch_size: int=32, max_len: int=32):
        super().__init__()
        self.file_path_dir = file_path_dir
        self.batch_size = batch_size
        self.max_len = max_len
        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            self.train_set = ToutiaoDataset(self.file_path_dir, "train", select_class=["100", "101"], max_len=self.max_len)
            self.eval_set = ToutiaoDataset(self.file_path_dir, "val", select_class=["100", "101"], max_len=self.max_len)
        if stage == "test" or stage is None:
            self.test_set = ToutiaoDataset(self.file_path_dir, "test", select_class=["100", "101"], max_len=self.max_len)
        if stage == "predict" or stage is None:
            self.predict_set = ToutiaoDataset

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.eval_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.predict_set, batch_size=self.batch_size, shuffle=False, num_workers=4)


class ToutiaoDataset(Dataset):
    def __init__(self, file_path_dir: str, mode: str, select_class: List[str],
                 padding: str="max_length", max_len: int=64, truncation: bool=True, label_map: Dict=None):
        self.file_path_dir = file_path_dir
        self.mode = mode
        self.padding = padding
        self.max_len = max_len
        self.truncation = truncation
        self.data = pd.read_csv(os.path.join(file_path_dir, f"{mode}/corpus-model-filtered-{mode}.csv"), sep="\t", header=0)
        self.select_data = self.data
        # self.select_data = pd.DataFrame(columns=self.data.columns)
        # for label_id in select_class:
        #     tmp = self.data[self.data.label_id == int(label_id)]
        #     self.select_data = pd.concat([self.select_data, tmp])
        # self.select_data = self.select_data.reset_index(drop=True)
        # del self.data
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')
        # print(self.tokenizer.pad_token_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if label_map is None:
            keys = [str(i) for i in range(100, 117)]
            values = range(0, 17)
            self.label_map = dict(zip(keys, values))
            del keys, values

    def compute_label(self, field):
        label = self.label_map[str(field["label_id"])]
        return np.array(label)

    def convert_str_to_id(self, field):
        text = field["text"]
        output = self.tokenizer(text, text_target=text, padding='max_length', truncation=self.truncation,
                               max_length=self.max_len, add_special_tokens=True)
        output["labels"] = output["labels"][1:]
        return output

    def __len__(self):
        return len(self.select_data)

    def __getitem__(self, item):
        field = self.select_data.loc[item]
        field = field.to_dict()
        feature = self.convert_str_to_id(field)
        # print(feature)
        return (
            np.array(feature["input_ids"]),
            np.array(feature["labels"]),
            np.array(feature["attention_mask"])
        )


class HuggingFaceHugeDatasetLoad(IterableDataset):
    def __init__(self, file_path_dir: str, mode: str, max_len):
        file_list = Path(file_path_dir).glob("*.csv")
        file_list = [file_path.name for file_path in file_list]
        total_len = len(file_list)
        tmp = {"train": [os.path.join(file_path_dir, raw) for raw in file_list[:int(total_len * 0.8)]],
               "validation": [os.path.join(file_path_dir, raw) for raw in file_list[int(total_len * 0.8):int(total_len * 0.9)]],
               "test": [os.path.join(file_path_dir, raw) for raw in file_list[int(total_len * 0.9):]]}
        self.data = load_dataset("csv", data_files=tmp, split=mode, streaming=True)
        self.max_len = max_len

    def __iter__(self):
        for field in self.data:
            tmp = field["input_ids"]
            process = list(map(int, re.sub("'|\[|\]", "", tmp).split(", ")))[:self.max_len]
            yield [np.array(process), np.array(process)[1:], np.ones([self.max_len])]


class PTMDataModules(LightningDataModule):
    def __init__(self, file_path_dir: str, batch_size: int=32, max_len: int=32):
        super().__init__()
        self.file_path_dir = file_path_dir
        self.batch_size = batch_size
        self.max_len = max_len
        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            self.train_set = HuggingFaceHugeDatasetLoad(self.file_path_dir, "train", max_len=self.max_len)
            self.eval_set = HuggingFaceHugeDatasetLoad(self.file_path_dir, "validation", max_len=self.max_len)
        if stage == "test" or stage is None:
            self.test_set = HuggingFaceHugeDatasetLoad(self.file_path_dir, "test", max_len=self.max_len)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_set, batch_size=self.batch_size)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.eval_set, batch_size=self.batch_size)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_set, batch_size=self.batch_size)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.predict_set, batch_size=self.batch_size)

#
if __name__ == "__main__":
    # dset = HuggingFaceHugeDatasetLoad(file_path_dir=r"d://workspace//Work//merge_corpus_csv", mode='train', max_len=512)
    # for i in dset:
    #     print(i)
    dm = PTMDataModules(file_path_dir=r"d://workspace//Work//merge_corpus_csv", max_len=32, batch_size=64)
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    for batch, _ in enumerate(train_loader):
        print(_)
