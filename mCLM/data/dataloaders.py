from sqlite3 import DateFromTicks
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

from sklearn.model_selection import train_test_split

from mCLM.tokenizer.molecule_tokenizer import MoleculeTokenizer

from mCLM.data.processing import insert_sublists, find_first_occurrence

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
        self, data, tokenizer, mol_tokenizer, trunc_length=512, block_to_idx=None, split=None,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.mol_tokenizer = mol_tokenizer
        self.MOL_start = tokenizer.convert_tokens_to_ids('[MOL]')
        self.MOL_end = tokenizer.convert_tokens_to_ids('[/MOL]')

        self._indices = None
        self.transform = None
        self.trunc_length = trunc_length

    def len(self):
        return len(self.data)

    def get(self, idx):

        d = self.data.iloc[idx]
        #print(d)

        raw_text = d['description']
        cleaned_text = d['cleaned_text']
        mol_list = d['mol_list']

        frags = [[self.mol_tokenizer.get_Idx(m) for m in mol.split('^')] for mol in mol_list]
        #print(frags)

        messages = [
            #{"role": "system", "content": "You are an expert chemist who designs molecules in a modular fashion or answers questions following the given instructions.",},
            {"role": "user", "content": "Please tell me a fact about a kinase inhibitor."},
            {"role": "assistant", "content": cleaned_text},
        ]
        message_chat = self.tokenizer.apply_chat_template(messages, tokenize=False)

        token_input = self.tokenizer(
            message_chat,
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
        token_input['attention_mask'] = token_input['attention_mask'].squeeze()
        token_input['input_ids'] = token_input['input_ids']#.unsqueeze(0)

        #print(token_input)
        #zz
        #for key in caption:
        #    caption[key] = caption[key].squeeze()

        rv = {
            "task_id": 0,
            "raw_text": raw_text,
            "input": {
                "input_ids": token_input['input_ids'],
                "labels": token_input['input_ids'],
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
    ):
        super().__init__()
        self.prepare_data_per_node = True

        self.base_model = base_model
        self.batch_size = batch_size
        self.trunc_length = trunc_length
        self.data_path = data_path
        self.config = config

    def setup(self, stage: str):
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token #llama3

        self.tokenizer.add_tokens(['[MOL]', '[/MOL]'])
        #model.resize_token_embeddings(len(tokenizer)) #put this somewhere
        
        start_idx = len(self.tokenizer)
        self.molecule_tokenizer = MoleculeTokenizer(start_idx)

        train_data = pd.read_csv(self.data_path + 'kinase_train.csv')#.head(2000)
        valid_data = pd.read_csv(self.data_path + 'kinase_valid.csv')#.head(1000)
        test_data = pd.read_csv(self.data_path + 'kinase_test.csv')#.head(1000)

        # FIXME: test only
        train_data = pd.read_csv(self.data_path + 'kinase_test.csv')
        valid_data = pd.read_csv(self.data_path + 'kinase_test.csv')

        #train_data[['mol_list', 'cleaned_text']] = train_data['description'].apply(extract_mol_content)
        train_data[['mol_list', 'cleaned_text']] = train_data['description'].progress_apply(lambda x: pd.Series(extract_mol_content(x)))
        valid_data[['mol_list', 'cleaned_text']] = valid_data['description'].progress_apply(lambda x: pd.Series(extract_mol_content(x)))
        test_data[['mol_list', 'cleaned_text']] = test_data['description'].progress_apply(lambda x: pd.Series(extract_mol_content(x)))

        #print(train_data)
        #zz

        #Preprocess molecule tokenizer
        block_to_idx = {}
        for df in [train_data, valid_data, test_data]:
            for d in df['mol_list']:
                for mol in d:
                    for block in mol.split('^'):
                        self.molecule_tokenizer.add_block(block)

        self.molecule_tokenizer.create_input()

        self.train_ds = KinaseDataset(
            train_data,
            self.tokenizer,
            self.molecule_tokenizer,
            split="train",
            trunc_length=self.trunc_length,
            block_to_idx = block_to_idx,
        )
        self.valid_ds = KinaseDataset(
            valid_data,
            self.tokenizer,
            self.molecule_tokenizer,
            split="valid",
            trunc_length=self.trunc_length,
            block_to_idx = block_to_idx,
        )
        self.test_ds = KinaseDataset(
            test_data,
            self.tokenizer,
            self.molecule_tokenizer,
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






class ConcatDataset(Dataset):
    """Combine two datasets to allow random interleaving of batches."""


    def __init__(self, datasets):
        self.datasets = datasets

        self.lens = [len(d) for d in datasets]
        self.bins = [sum(self.lens[:i]) for i in range(len(self.lens))]
        #print(self.lens, self.bins)

    def __len__(self):
        return sum(self.lens)

    def __getitem__(self, idx):
        
        didx = np.digitize(idx, self.bins) - 1 #not zero indexed for some reason
        #print(idx, didx)
        return self.datasets[didx].__getitem__(idx - self.bins[didx])


class GeneralDataset(Dataset):

    def __init__(
        self, data, tokenizer, mol_tokenizer, task_name, trunc_length=512,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.mol_tokenizer = mol_tokenizer
        self.MOL_start = tokenizer.convert_tokens_to_ids('[MOL]')
        self.MOL_end = tokenizer.convert_tokens_to_ids('[/MOL]')

        self._indices = None
        self.transform = None
        self.trunc_length = trunc_length
        self.task_name = task_name

    def len(self):
        return len(self.data)

    def get(self, idx):

        d = self.data.iloc[idx]

        raw_instruction = d['instruction']
        raw_response = d['response']
        cleaned_instruction = d['cleaned_instruction']
        cleaned_response = d['cleaned_response']
        mol_list = d['mol_list']

        frags = [[self.mol_tokenizer.get_Idx(m) for m in mol.split('^')] for mol in mol_list]

        messages = [
            #{"role": "system", "content": "You are an expert chemist who designs molecules in a modular fashion or answers questions following the given instructions.",},
            {"role": "user", "content": cleaned_instruction},
            {"role": "assistant", "content": cleaned_response},
        ]
        message_chat = self.tokenizer.apply_chat_template(messages, tokenize=False)

        token_input = self.tokenizer(
            message_chat,
            truncation=True,
            max_length=self.trunc_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        token_input['input_ids'] = torch.Tensor(insert_sublists(token_input['input_ids'].squeeze(), frags, self.MOL_start, self.MOL_end)[:self.trunc_length]).to(torch.int)#, dtype=torch.int32)
        pad_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        num_attn = find_first_occurrence(token_input['input_ids'], pad_id)
        token_input['attention_mask'][:,:num_attn] = 1
        token_input['attention_mask'] = token_input['attention_mask'].squeeze()
        token_input['input_ids'] = token_input['input_ids']

        rv = {
            "task_id": self.task_name,
            "raw_instruction": raw_instruction,
            "raw_response": raw_response,
            "input": {
                "input_ids": token_input['input_ids'],
                "labels": token_input['input_ids'],
                "attention_mask": token_input['attention_mask'],
            },
        }
        return rv


class MolInstDataset(Dataset):

    def __init__(
        self, data, tokenizer, mol_tokenizer, task_name, trunc_length=512,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.mol_tokenizer = mol_tokenizer
        self.MOL_start = tokenizer.convert_tokens_to_ids('[MOL]')
        self.MOL_end = tokenizer.convert_tokens_to_ids('[/MOL]')

        self._indices = None
        self.transform = None
        self.trunc_length = trunc_length
        self.task_name = task_name

    def len(self):
        return len(self.data)

    def get(self, idx):

        d = self.data.iloc[idx]

        instruction = d['instruction']
        inp = d['input']
        output = d['output']

        messages = [
            #{"role": "system", "content": "You are an expert chemist who designs molecules in a modular fashion or answers questions following the given instructions.",},
            {"role": "user", "content": instruction + '\n\n' + inp},
            {"role": "assistant", "content": output},
        ]
        message_chat = self.tokenizer.apply_chat_template(messages, tokenize=False)

        token_input = self.tokenizer(
            message_chat,
            truncation=True,
            max_length=self.trunc_length,
            padding="max_length",
            return_tensors="pt",
        )
        

        rv = {
            "task_id": self.task_name,
            "input": {
                "input_ids": token_input['input_ids'],
                "labels": token_input['input_ids'],
                "attention_mask": token_input['attention_mask'],
            },
        }
        return rv


class TuluDataset(Dataset):

    def __init__(
        self, data, tokenizer, mol_tokenizer, task_name, trunc_length=512,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.mol_tokenizer = mol_tokenizer
        self.MOL_start = tokenizer.convert_tokens_to_ids('[MOL]')
        self.MOL_end = tokenizer.convert_tokens_to_ids('[/MOL]')

        self._indices = None
        self.transform = None
        self.trunc_length = trunc_length
        self.task_name = task_name

    def len(self):
        return len(self.data)

    def get(self, idx):

        d = self.data.iloc[idx]

        messages = eval(d['messages'].replace("'}\n {'", "'},{'"))

        message_chat = self.tokenizer.apply_chat_template(messages, tokenize=False)

        token_input = self.tokenizer(
            message_chat,
            truncation=True,
            max_length=self.trunc_length,
            padding="max_length",
            return_tensors="pt",
        )
        

        rv = {
            "task_id": self.task_name,
            "input": {
                "input_ids": token_input['input_ids'],
                "labels": token_input['input_ids'],
                "attention_mask": token_input['attention_mask'],
            },
        }
        return rv

def extract_mol_content2(instruction, response):
    try:
        ml1, clean_instruction = extract_mol_content(instruction)
    except:
        print(instruction)
        zz
    try:
        ml2, clean_response = extract_mol_content(response)
    except:
        print(response)
        zz
    return ml1 + ml2, clean_instruction, clean_response

class TotalDataModule(LightningDataModule):
    def __init__(
        self,
        config,
        base_model,
        batch_size=4,
        trunc_length=512,
        instruction_data_path="captions/",
        synthetic_data_path="captions/",
    ):
        super().__init__()
        self.prepare_data_per_node = True

        self.base_model = base_model
        self.batch_size = batch_size
        self.trunc_length = trunc_length
        self.instruction_data_path = instruction_data_path
        self.synthetic_data_path = synthetic_data_path
        self.config = config
        self.seed = config['seed']

    def setup(self, stage: str):
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token #llama3

        self.tokenizer.add_tokens(['[MOL]', '[/MOL]'])
        #model.resize_token_embeddings(len(tokenizer)) #put this somewhere
        
        start_idx = len(self.tokenizer)
        self.molecule_tokenizer = MoleculeTokenizer(start_idx)

        to_split_data = []
        train_data = []
        valid_data = []
        test_data = []

        
        '''
        for subdir in ['pos_neg', 'pos_pos', 'property_to_mol','multi_property_to_mol', 'mol_only','mCLM','regression', 'classification']:
            ddir = osp.join(self.synthetic_data_path, subdir)
            files = [f for f in os.listdir(ddir) if os.path.isfile(os.path.join(ddir, f))]
            for f in files:
                print(f)
                df = pd.read_csv(osp.join(ddir, f), dtype={'instruction': str, 'response': str})
                print(df)
                if len(df) == 0: continue
                df[['mol_list', 'cleaned_instruction', 'cleaned_response']] = df.progress_apply(lambda x: pd.Series(extract_mol_content2(x['instruction'], x['response'])), axis=1)
                to_split_data.append((df, f.replace('.csv', '')))
                #if f == 'Tox21_class.csv': break
        '''

        ddir = osp.join(self.instruction_data_path)
        f = 'tulu-3-sft_train.csv'
        df = pd.read_csv(osp.join(ddir, f), dtype={'messages': str})
        print(f)
        print(df)
        to_split_data.append((df, f.replace('.csv', '')))
        
        ddir = osp.join(self.instruction_data_path)
        f = 'mol-inst_biomol_text_train.csv'
        df = pd.read_csv(osp.join(ddir, f), dtype={'instruction': str, 'input': str, 'output': str})
        print(f)
        print(df)
        train_data.append((df, f.replace('.csv', '')))
        
        ddir = osp.join(self.instruction_data_path)
        f = 'mol-inst_biomol_text_test.csv'
        df = pd.read_csv(osp.join(ddir, f), dtype={'instruction': str, 'input': str, 'output': str})
        print(f)
        print(df)
        valid_data.append((df, f.replace('.csv', '')))
        test_data.append((df, f.replace('.csv', '')))

        #ddir = osp.join(self.instruction_data_path)
        #f = 'SMolInstruct_train.csv'
        #df = pd.read_csv(osp.join(ddir, f), dtype={'instruction': str, 'response': str})
        #print(f)
        #print(df)
        #df[['mol_list', 'cleaned_instruction', 'cleaned_response']] = df.progress_apply(lambda x: pd.Series(extract_mol_content2(x['instruction'], x['response'])), axis=1)
        #train_data.append((df, f.replace('.csv', '')))

        ddir = osp.join(self.instruction_data_path)
        f = 'SMolInstruct_val.csv'
        df = pd.read_csv(osp.join(ddir, f), dtype={'instruction': str, 'response': str})
        print(f)
        print(df)
        df[['mol_list', 'cleaned_instruction', 'cleaned_response']] = df.progress_apply(lambda x: pd.Series(extract_mol_content2(x['instruction'], x['response'])), axis=1)
        valid_data.append((df, f.replace('.csv', '')))

        ddir = osp.join(self.instruction_data_path)
        f = 'SMolInstruct_test.csv'
        df = pd.read_csv(osp.join(ddir, f), dtype={'instruction': str, 'response': str})
        print(f)
        print(df)
        df[['mol_list', 'cleaned_instruction', 'cleaned_response']] = df.progress_apply(lambda x: pd.Series(extract_mol_content2(x['instruction'], x['response'])), axis=1)
        test_data.append((df, f.replace('.csv', '')))
        

        # FIXME: test only
        #train_data = pd.read_csv(self.data_path + 'kinase_test.csv')
        #valid_data = pd.read_csv(self.data_path + 'kinase_test.csv')

        #Preprocess molecule tokenizer
        block_to_idx = {}
        for dfs in [to_split_data, train_data, valid_data, test_data]:
            for df, task in dfs:
                if 'mol_list' in df:
                    for d in df['mol_list']:
                        for mol in d:
                            for block in mol.split('^'):
                                #if block == '.CCCCCCCCCCCCOS(=O)(=O)O': print(task, mol)
                                self.molecule_tokenizer.add_block(block)

        self.molecule_tokenizer.create_input()

        self.train_dses = []
        self.valid_dses = []
        self.test_dses = []

        for df, task in to_split_data:
            ts = min(200, max(int(0.01*len(df)), 10))
            train_df, val_df = train_test_split(df, test_size=ts, random_state = self.seed)
            val_df, test_df = train_test_split(df, test_size=0.5, random_state = self.seed)
            if task.startswith('tulu'): ds_type = TuluDataset
            elif task.startswith('mol-inst'): ds_type = MolInstDataset
            else: ds_type = GeneralDataset
            ds = ds_type(
                train_df,
                self.tokenizer,
                self.molecule_tokenizer,
                task_name=task,
                trunc_length=self.trunc_length,
            )
            self.train_dses.append(ds)
            ds = ds_type(
                val_df,
                self.tokenizer,
                self.molecule_tokenizer,
                task_name=task,
                trunc_length=self.trunc_length,
            )
            self.valid_dses.append(ds)
            ds = ds_type(
                test_df,
                self.tokenizer,
                self.molecule_tokenizer,
                task_name=task,
                trunc_length=self.trunc_length,
            )
            self.test_dses.append(ds)
            

        for df, task in train_data:
            if task.startswith('tulu'): ds_type = TuluDataset
            elif task.startswith('mol-inst'): ds_type = MolInstDataset
            else: ds_type = GeneralDataset
            ds = ds_type(
                df,
                self.tokenizer,
                self.molecule_tokenizer,
                task_name=task,
                trunc_length=self.trunc_length,
            )
            self.train_dses.append(ds)

        self.train_ds = ConcatDataset(self.train_dses)

        for df, task in valid_data:
            ds = GeneralDataset(
                df,
                self.tokenizer,
                self.molecule_tokenizer,
                task_name=task,
                trunc_length=self.trunc_length,
            )
            self.valid_dses.append(ds)

        for df, task in test_data:
            ds = GeneralDataset(
                df,
                self.tokenizer,
                self.molecule_tokenizer,
                task_name=task,
                trunc_length=self.trunc_length,
            )
            self.test_dses.append(ds)


    def train_dataloader(self):
        return CustomDataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return [CustomDataLoader(
            ds, batch_size=self.batch_size, num_workers=0, shuffle=False
        ) for ds in self.valid_dses]

    def test_dataloader(self):
        return [CustomDataLoader(
            ds, batch_size=self.batch_size, num_workers=0, shuffle=False
        ) for ds in self.test_dses]

    def teardown(self, stage: str):
        pass






class SMolInstructDataModule(LightningDataModule):
    def __init__(
        self,
        config,
        base_model,
        batch_size=4,
        trunc_length=512,
        instruction_data_path="captions/",
        synthetic_data_path="captions/",
    ):
        super().__init__()
        self.prepare_data_per_node = True

        self.base_model = base_model
        self.batch_size = batch_size
        self.trunc_length = trunc_length
        self.instruction_data_path = instruction_data_path
        self.synthetic_data_path = synthetic_data_path
        self.config = config
        self.seed = config['seed']

    def setup(self, stage: str):
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token #llama3

        self.tokenizer.add_tokens(['[MOL]', '[/MOL]'])
        #model.resize_token_embeddings(len(tokenizer)) #put this somewhere
        
        start_idx = len(self.tokenizer)
        self.molecule_tokenizer = MoleculeTokenizer(start_idx)

        train_data = []
        valid_data = []
        test_data = []

        ddir = osp.join(self.instruction_data_path)
        f = 'SMolInstruct_train.csv'
        df = pd.read_csv(osp.join(ddir, f), dtype={'instruction': str, 'response': str})
        print(f)
        print(df)
        df[['mol_list', 'cleaned_instruction', 'cleaned_response']] = df.progress_apply(lambda x: pd.Series(extract_mol_content2(x['instruction'], x['response'])), axis=1)
        train_data.append((df, f.replace('.csv', '')))

        ddir = osp.join(self.instruction_data_path)
        f = 'SMolInstruct_val.csv'
        df = pd.read_csv(osp.join(ddir, f), dtype={'instruction': str, 'response': str})
        print(f)
        print(df)
        df[['mol_list', 'cleaned_instruction', 'cleaned_response']] = df.progress_apply(lambda x: pd.Series(extract_mol_content2(x['instruction'], x['response'])), axis=1)
        valid_data.append((df, f.replace('.csv', '')))

        ddir = osp.join(self.instruction_data_path)
        f = 'SMolInstruct_test.csv'
        df = pd.read_csv(osp.join(ddir, f), dtype={'instruction': str, 'response': str})
        print(f)
        print(df)
        df[['mol_list', 'cleaned_instruction', 'cleaned_response']] = df.progress_apply(lambda x: pd.Series(extract_mol_content2(x['instruction'], x['response'])), axis=1)
        test_data.append((df, f.replace('.csv', '')))

        #Preprocess molecule tokenizer
        block_to_idx = {}
        for dfs in [train_data, valid_data, test_data]:
            for df, task in dfs:
                for d in df['mol_list']:
                    for mol in d:
                        for block in mol.split('^'):
                            #if block == '.CCCCCCCCCCCCOS(=O)(=O)O': print(task, mol)
                            self.molecule_tokenizer.add_block(block)

        self.molecule_tokenizer.create_input()

        self.train_dses = []
        self.valid_dses = []
        self.test_dses = []

        for df, task in train_data:
            ds = GeneralDataset(
                DateFromTicks,
                self.tokenizer,
                self.molecule_tokenizer,
                task_name=task,
                trunc_length=self.trunc_length,
            )
            self.train_dses.append(ds)
        self.train_ds = ConcatDataset(self.train_dses)

        for df, task in valid_data:
            ds = GeneralDataset(
                df,
                self.tokenizer,
                self.molecule_tokenizer,
                task_name=task,
                trunc_length=self.trunc_length,
            )
            self.valid_dses.append(ds)

        for df, task in test_data:
            ds = GeneralDataset(
                df,
                self.tokenizer,
                self.molecule_tokenizer,
                task_name=task,
                trunc_length=self.trunc_length,
            )
            self.test_dses.append(ds)
        #zz

    def train_dataloader(self):
        return CustomDataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return [CustomDataLoader(
            ds, batch_size=self.batch_size, num_workers=0, shuffle=False
        ) for ds in self.valid_dses]

    def test_dataloader(self):
        return [CustomDataLoader(
            ds, batch_size=self.batch_size, num_workers=0, shuffle=False
        ) for ds in self.test_dses]

    def teardown(self, stage: str):
        pass

