import lightning as L
from lightning import LightningDataModule
import sys
import torch
import torch.utils.data
from mCLM.data.processing import smiles_to_data, smiles_list_to_mols
from collections.abc import Mapping
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter
from transformers import AutoTokenizer
from typing import Any, List, Optional, Sequence, Union
from tqdm import tqdm

import pandas as pd

from sklearn.model_selection import train_test_split

from rdkit.Chem.inchi import MolToInchi
from rdkit import Chem

import numpy as np

from collections import defaultdict
from sklearn.utils import resample

import random

import os
import os.path as osp

import pickle

from MolCapArena.data.processing import (
    MolecularDataset,
    index_predetermined_split,
    MolecularSubset,
)


def canonicalize(smi):
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    except:
        return None


class CustomCollater:
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ):
        self.dataset = dataset
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch: List[Any]) -> Any:
        elem = batch[0]
        if isinstance(elem, Batch):

            rv = [b.to_data_list() for b in batch]
            rv = [item for sublist in rv for item in sublist]
            return Batch.from_data_list(rv)
        elif isinstance(elem, BaseData):
            return Batch.from_data_list(
                batch,
                follow_batch=self.follow_batch,
                exclude_keys=self.exclude_keys,
            )
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        # elif isinstance(elem, TensorFrame):
        #    return torch_frame.cat(batch, along='row')
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, "_fields"):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f"DataLoader found invalid type: '{type(elem)}'")

    def collate_fn(self, batch: List[Any]) -> Any:
        # if isinstance(self.dataset, OnDiskDataset):
        #    return self(self.dataset.multi_get(batch))
        return self(batch)


class CustomDataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """

    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        kwargs.pop("collate_fn", None)

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        self.collator = CustomCollater(dataset, follow_batch, exclude_keys)

        # if isinstance(dataset, OnDiskDataset):
        #    dataset = range(len(dataset))

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=self.collator.collate_fn,
            **kwargs,
        )



class KinaseDataset(Dataset):

    def __init__(
        self, data, tokenizer, trunc_length=512, split=None, caption_data=None
    ):
        self.data = data
        self.tokenizer = tokenizer

        self._indices = None
        self.transform = None
        self.trunc_length = trunc_length
        self.caption_data = caption_data

    def len(self):
        return len(self.data)

    def get(self, idx):
        SMILES = self.data.smiles[idx]
        y = self.data.y[idx]

        act = torch.tensor(y, dtype=float).float()

        molecule_input = smiles_to_data(SMILES)

        if self.caption_data is not None:
            text_caption = self.caption_data.loc[SMILES]["captions"]
            if type(text_caption) != str:  # it's a blank
                text_caption = ""
            caption = self.tokenizer(
                text_caption,
                truncation=True,
                max_length=self.trunc_length,
                padding="max_length",
                return_tensors="pt",
            )
            for key in caption:
                caption[key] = caption[key].squeeze()
        else:
            text_caption = caption = "None"

        rv = {
            "SMILES": SMILES,
            "SMILES": SMILES,
            "task_id": 0,
            "caption": text_caption,
            "input": {
                "molecule": molecule_input,
                "activity": act,
                "caption": caption,
            },
        }
        return rv



class KinaseDataModule(LightningDataModule):
    def __init__(
        self,
        config,
        pretrained_text_model,
        batch_size=4,
        trunc_length=512,
        fold_idx=0,
        data_path="captions/",
        caption_source=None,
    ):
        super().__init__()
        self.prepare_data_per_node = True

        self.pretrained_text_model = pretrained_text_model
        self.batch_size = batch_size
        self.trunc_length = trunc_length
        self.data_path = data_path
        self.config = config
        self.fold_idx = fold_idx
        self.caption_source = caption_source

    def setup(self, stage: str):
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_text_model)

        self.train_ds = KinaseDataset(
            self.tokenizer,
            split="train",
            trunc_length=self.trunc_length,
            caption_data=caption_data,
        )
        self.valid_ds = KinaseDataset(
            self.tokenizer,
            split="valid",
            trunc_length=self.trunc_length,
            caption_data=caption_data,
        )
        self.test_ds = KinaseDataset(
            self.tokenizer,
            split="test",
            trunc_length=self.trunc_length,
            caption_data=caption_data,
        )

    def train_dataloader(self):
        return CustomDataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return CustomDataLoader(
            self.valid_ds, batch_size=self.batch_size, num_workers=0, shuffle=False
        )

    def test_dataloader(self):
        return CustomDataLoader(
            self.test_ds, batch_size=self.batch_size, num_workers=0, shuffle=False
        )

    def teardown(self, stage: str):
        pass

