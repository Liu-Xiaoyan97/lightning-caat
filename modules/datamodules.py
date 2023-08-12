#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/7/10 15:28
# @Author  : lxy15058247683@aliyun.com
# @FileName: datamodules.py
# @Copyright: MIT
import json
import os
from optparse import Option

import numpy as np
import pandas
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional, Dict
from transformers import PreTrainedTokenizer, AutoTokenizer
import platform
num_workers = 0
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
        self.select_data = pd.DataFrame(columns=self.data.columns)
        for label_id in select_class:
            tmp = self.data[self.data.label_id == int(label_id)]
            self.select_data = pd.concat([self.select_data, tmp])
        self.select_data = self.select_data.reset_index(drop=True)
        del self.data
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


if __name__ == "__main__":
    dm = DataModules(file_path_dir=r"D:\workspace\Work\lightning-caat\Data\zh")
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    for batch, _ in enumerate(train_loader):
        pass
