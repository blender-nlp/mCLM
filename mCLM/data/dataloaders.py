import lightning as L
from lightning import LightningDataModule
import sys
import torch
import torch.utils.data
from mCLM.data.processing import smiles_to_data, smiles_list_to_mols, extract_mol_content
from collections.abc import Mapping
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter
from transformers import AutoTokenizer
from typing import Any, List, Optional, Sequence, Union

import pandas as pd

from tqdm import tqdm
tqdm.pandas()

from sklearn.model_selection import train_test_split

from rdkit import Chem

import numpy as np

from collections import defaultdict
from sklearn.utils import resample

import random

import os
import os.path as osp

import pickle

from mCLM.data.processing import (
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


def insert_sublists(main_list, sublists, start=2, end=4):
    result = []
    sublist_index = 0  # Track which sublist to insert

    i = 0
    while i < len(main_list):
        result.append(main_list[i])
        if main_list[i] == start and i + 1 < len(main_list) and main_list[i + 1] == end:
            # Insert the next sublist
            if sublist_index < len(sublists):
                result.extend(sublists[sublist_index])
                sublist_index += 1
                result.append(main_list[i+1])
            i += 1  # Skip the next element (end)
        i += 1

    return result

def find_first_occurrence(tensor, num):
    indices = torch.where(tensor == num)[0]  # Get indices where tensor equals num
    return indices[0].item() if indices.numel() > 0 else len(tensor)


class KinaseDataset(Dataset):

    def __init__(
        self, data, tokenizer, trunc_length=512, block_to_idx=None, split=None,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.MOL_start = tokenizer.convert_tokens_to_ids('[MOL]')
        self.MOL_end = tokenizer.convert_tokens_to_ids('[/MOL]')

        self._indices = None
        self.transform = None
        self.trunc_length = trunc_length
        self.block_to_idx = block_to_idx

    def len(self):
        return len(self.data)

    def get(self, idx):

        d = self.data.iloc[idx]
        #print(d)

        raw_text = d['description']
        cleaned_text = d['cleaned_text']
        mol_list = d['mol_list']

        frags = [[self.block_to_idx[m] for m in mol.split('^')] for mol in mol_list]
        #print(frags)

        token_input = self.tokenizer(
            cleaned_text,
            truncation=True,
            max_length=self.trunc_length,
            padding="max_length",
            return_tensors="pt",
        )

        #print(token_input)

        token_input['input_ids'] = torch.Tensor(insert_sublists(token_input['input_ids'].squeeze(), frags, self.MOL_start, self.MOL_end)[:self.trunc_length]).to(torch.int)#, dtype=torch.int32)
        pad_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        #print(token_input['input_ids'], pad_id)
        num_attn = find_first_occurrence(token_input['input_ids'], pad_id)
        token_input['attention_mask'][:,:num_attn] = 1
        token_input['input_ids'] = token_input['input_ids'].unsqueeze(0)

        #print(token_input)
        #zz
        #for key in caption:
        #    caption[key] = caption[key].squeeze()

        rv = {
            "task_id": 0,
            "raw_text": raw_text,
            "input": {
                "input_ids": token_input['input_ids'],
                "attention_mask": token_input['attention_mask'],
            },
        }
        return rv



class KinaseDataModule(LightningDataModule):
    def __init__(
        self,
        config,
        base_model,
        batch_size=4,
        trunc_length=512,
        data_path="captions/",
        molecule_tokenizer= None,
    ):
        super().__init__()
        self.prepare_data_per_node = True

        self.base_model = base_model
        self.batch_size = batch_size
        self.trunc_length = trunc_length
        self.data_path = data_path
        self.config = config
        self.molecule_tokenizer = molecule_tokenizer

    def setup(self, stage: str):
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token #llama3

        self.tokenizer.add_tokens(['[MOL]', '[/MOL]'])
        #model.resize_token_embeddings(len(tokenizer)) #put this somewhere

        train_data = pd.read_csv(self.data_path + 'kinase_train.csv')
        valid_data = pd.read_csv(self.data_path + 'kinase_valid.csv')
        test_data = pd.read_csv(self.data_path + 'kinase_test.csv')

        # FIXME: test only
        train_data = pd.read_csv(self.data_path + 'kinase_test.csv')
        valid_data = pd.read_csv(self.data_path + 'kinase_test.csv')

        #train_data[['mol_list', 'cleaned_text']] = train_data['description'].apply(extract_mol_content)
        train_data[['mol_list', 'cleaned_text']] = train_data['description'].progress_apply(lambda x: pd.Series(extract_mol_content(x)))
        valid_data[['mol_list', 'cleaned_text']] = valid_data['description'].progress_apply(lambda x: pd.Series(extract_mol_content(x)))
        test_data[['mol_list', 'cleaned_text']] = test_data['description'].progress_apply(lambda x: pd.Series(extract_mol_content(x)))

        #print(train_data)
        #zz

        start_idx = len(self.tokenizer)
        block_to_idx = {}
        for df in [train_data, valid_data, test_data]:
            for d in df['mol_list']:
                for mol in d:
                    for block in mol.split('^'):
                        if block not in block_to_idx:
                            block_to_idx[block] = start_idx + len(block_to_idx)

        #print(block_to_idx)
        #zz
        #if False:
        self.GNN_input_map = {}
        for block in tqdm(block_to_idx, desc='Creating GNN Input'):
            self.GNN_input_map[block_to_idx[block]] = smiles_to_data(block)

        self.train_ds = KinaseDataset(
            train_data,
            self.tokenizer,
            split="train",
            trunc_length=self.trunc_length,
            block_to_idx = block_to_idx,
        )
        self.valid_ds = KinaseDataset(
            valid_data,
            self.tokenizer,
            split="valid",
            trunc_length=self.trunc_length,
            block_to_idx = block_to_idx,
        )
        self.test_ds = KinaseDataset(
            test_data,
            self.tokenizer,
            split="test",
            trunc_length=self.trunc_length,
            block_to_idx = block_to_idx,
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

