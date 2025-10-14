from sqlite3 import DateFromTicks
import lightning as L
from lightning import LightningDataModule
import sys
import torch
import torch.utils.data
from torch.utils.data import ConcatDataset
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
# Chi Debug
# from torchdata.stateful_dataloader import StatefulDataLoader
from torch.utils.data import DataLoader

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

import ast

import os
import os.path as osp

import pickle

from mCLM.data.processing import (
    MolecularDataset,
    index_predetermined_split,
    MolecularSubset,
)


def replace_ends(mol_list):

    new_mol_list = []
    for mol in mol_list:
        new_react = []
        for mo in mol.split('.'):

            split = mo.split('^')
            new_mol = []
            for m in split:
                if ('[1*]' in m) and (not '[2*]' in m): new_mol.append(m.replace('[1*]', '[3*]'))
                elif (not '[1*]' in m) and ('[2*]' in m): new_mol.append(m.replace('[2*]', '[3*]'))
                else: new_mol.append(m)

            new_react.append('^'.join(new_mol))
        new_mol_list.append('.'.join(new_react))

    return new_mol_list


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

# Chi Debug
# class CustomDataLoader(StatefulDataLoader):
class CustomDataLoader(DataLoader):
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





class StatefulShuffleDataset(Dataset):
    """Shuffle a dataset without using shuffle=True because lightning doesn't support Random shuffler with fault tolerance."""

    def __init__(self, dataset, seed):
        self.dataset = dataset
        self._indices = None
        self.transform = None
        self.seed(seed)

    def seed(self, seed):
        np.random.seed(seed)
        self.random = np.arange(len(self.dataset))
        np.random.shuffle(self.random)

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        return self.dataset[self.random[idx]]



class GeneralDataset(Dataset):

    def __init__(
        self, data, tokenizer, mol_tokenizer, task_name,trunc_length=512, shrink_size=None, only_mol=False,
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
        self.shrink_size = shrink_size

        self.only_mol = only_mol

        if self.shrink_size != None:
            self.all_data = data
            self.data = self.all_data.sample(min(self.shrink_size, len(self.all_data)), random_state=0)

    def set_new_epoch(self, epoch):
        if self.shrink_size != None:
            self.data = self.all_data.sample(min(self.shrink_size, len(self.all_data)), random_state=epoch)


    def len(self):
        return len(self.data)

    def get(self, idx):

        d = self.data.iloc[idx]

        #raw_instruction = d['instruction']
        #raw_response = d['response']
        cleaned_instruction = d['cleaned_instruction'].replace('[MOL] [/MOL]', '[MOL][/MOL]')
        cleaned_response = d['cleaned_response'].replace('[MOL] [/MOL]', '[MOL][/MOL]')
        mol_list = d['mol_list']

        pad_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)

        #frags = [[self.mol_tokenizer.get_Idx(m) for m in mol.split('^')] for mol in mol_list]
        #self.mol_tokenizer.create_input_from_list(sum([mol.split('^') for mol in mol_list], []))

        messages = [
            #{"role": "system", "content": "You are the mCLM, a helpful expert chemist who designs molecules in a modular fashion or answers questions.",},
            {"role": "user", "content": cleaned_instruction},
            {"role": "assistant", "content": cleaned_response},
        ]

        if self.only_mol:

            rv = {
                "task_id": self.task_name,
                "raw_instruction": cleaned_instruction,
                "raw_response": cleaned_response,
                "mol_list": str(mol_list),        
                'messages':messages,
            }
            return rv


        rv = {
            "task_id": self.task_name,
            "raw_instruction": cleaned_instruction,
            "raw_response": cleaned_response,
            "mol_list": str(mol_list),
            'messages':messages,
        }
        return rv


class MolInstDataset(Dataset):

    def __init__(
        self, data, tokenizer, mol_tokenizer, task_name, trunc_length=512, shrink_size=None,
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

    def set_new_epoch(self, epoch):
        pass

    def len(self):
        return len(self.data)

    def get(self, idx):

        d = self.data.iloc[idx]

        instruction = d['instruction']
        inp = d['input']
        output = d['output']

        raw_instruction = instruction + '\n\n' + inp.strip()
        raw_response = output

        messages = [
            #{"role": "system", "content": "You are the mCLM, a helpful expert chemist who designs molecules in a modular fashion or answers questions.",},
            {"role": "user", "content": instruction + '\n\n' + inp.strip()},
            {"role": "assistant", "content": output},
        ]
        

        rv = {
            "task_id": self.task_name,
            "raw_instruction": raw_instruction,
            "raw_response": raw_response,
            "mol_list": '',
            'messages':messages,
        }
        return rv


class TuluDataset(Dataset):

    def __init__(
        self, data, tokenizer, mol_tokenizer, task_name, trunc_length=512, shrink_size=None,
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

    def set_new_epoch(self, epoch):
        pass

    def len(self):
        return len(self.data)

    def get(self, idx):

        d = self.data.iloc[idx]

        #messages = eval(d['messages'].replace("'}\n {'", "'},{'"))
        raw_instruction = d['messages'][0]['content']
        try:
            raw_response = d['messages'][1]['content']
        except: raw_response = ""
        messages = d['messages']



        rv = {
            "task_id": self.task_name,
            "raw_instruction": raw_instruction,
            "raw_response": raw_response,
            "mol_list": '',
            'messages':messages,
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
        general_data_path="/shared/nas/data/m1/shared-resource/MoleculeLanguage/mCLM/instruction/processed/",
        synthetic_data_path="captions/",
        GNN_cache = '../GNN_input_cache/Total.molecule_tokenizer.v4.pth',
        shrink_data = None,
    ):
        super().__init__()
        self.prepare_data_per_node = True

        self.base_model = base_model
        self.batch_size = batch_size
        self.trunc_length = trunc_length
        self.instruction_data_path = instruction_data_path
        self.synthetic_data_path = synthetic_data_path
        self.general_data_path = general_data_path
        self.config = config
        self.seed = config['seed']
        self.GNN_cache = GNN_cache
        self.shrink_data = shrink_data

    def set_new_epoch(self, epoch):
        for ds in self.train_dses:
            ds.set_new_epoch(epoch)
        for ds in self.valid_dses:
            ds.set_new_epoch(epoch)
        for ds in self.test_dses:
            ds.set_new_epoch(epoch)

    def setup(self, stage: str):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['pretrained_tokenizer'])
        self.tokenizer.pad_token = self.tokenizer.eos_token #llama3

        self.tokenizer.add_tokens(['[MOL]', '[/MOL]'])
        #model.resize_token_embeddings(len(tokenizer)) #put this somewhere

        start_idx = len(self.tokenizer)

        to_split_data = []
        train_data = []
        valid_data = []
        test_data = []

        print('Loading Data')

        for subdir in ['synthetic_chembl', 'synthetic_admet_chembl', 'pos_neg', 'pos_pos', 'property_to_mol','multi_property_to_mol', 'mol_only','mCLM','regression', 'classification']:
            ddir = osp.join(self.synthetic_data_path, subdir)
            print(ddir)
            files = [f for f in os.listdir(ddir) if os.path.isfile(os.path.join(ddir, f))]
            total = 0
            for f in files:
                print(f)
                try:
                    df = pd.read_csv(osp.join(ddir, f), dtype={'instruction': str, 'response': str, 'cleaned_instruction': str, 'cleaned_response': str},keep_default_na=False,na_values=[])
                except pd.errors.EmptyDataError:
                    print(f"Warning: {f} is empty. Skipping.")
                    continue
                if len(df) == 0: continue
                df['mol_list'] = df['mol_list'].apply(ast.literal_eval)

                #if self.shrink_data != None:
                #    df = df.sample(min(self.shrink_data, len(df)))
                print(len(df))
                total += len(df)
                #df[['mol_list', 'cleaned_instruction', 'cleaned_response']] = df.progress_apply(lambda x: pd.Series(extract_mol_content2(x['instruction'], x['response'])), axis=1)
                
                if subdir == 'synthetic_admet_chembl': task_name = subdir.replace('synthetic_', 'oracle/') + '/' + f.replace('.csv', '')
                elif subdir == 'synthetic_chembl': task_name = subdir.replace('synthetic_', '') + '/' + f.replace('.csv', '')
                elif 'synthetic' in f and subdir == 'mCLM': task_name = subdir.replace('mCLM', 'oracle') + '/' + f.replace('.csv', '').replace('_synthetic', '')
                elif subdir == 'mCLM': task_name = subdir.replace('mCLM', 'TDC') + '/' + f.replace('.csv', '')
                else: task_name = subdir + '/' + f.replace('.csv', '')
                
                
                to_split_data.append((df, task_name, self.shrink_data))
                #if f == 'Tox21_class.csv': break
            print(subdir, total)

        ddir = osp.join(self.general_data_path)
        f = 'tulu-3-sft_train.csv'
        print(f)
        df = pd.read_csv(osp.join(ddir, f),keep_default_na=False,na_values=[])
        if self.config['downsample_tulu'] != None:
            df = df.sample(frac=self.config['downsample_tulu'], random_state=self.seed)
        df['messages'] = df['messages'].apply(lambda x : x.replace("'}\n {'", "'},{'").replace("'} {'", "'},{'")).apply(ast.literal_eval)
        print(len(df))
        to_split_data.append((df, f.replace('.csv', ''),None))

        ddir = osp.join(self.general_data_path)
        f = 'mol-inst_biomol_text_train.csv'
        print(f)
        df = pd.read_csv(osp.join(ddir, f), dtype={'instruction': str, 'input': str, 'output': str},keep_default_na=False,na_values=[])
        print(len(df))
        train_data.append((df, f.replace('.csv', ''),None))

        ddir = osp.join(self.general_data_path)
        f = 'mol-inst_biomol_text_test.csv'
        print(f)
        df = pd.read_csv(osp.join(ddir, f), dtype={'instruction': str, 'input': str, 'output': str},keep_default_na=False,na_values=[])
        print(len(df))
        valid_data.append((df, f.replace('.csv', ''),None))
        test_data.append((df, f.replace('.csv', ''),None))

        ddir = osp.join(self.instruction_data_path)
        f = 'SMolInstruct_train.csv'
        print(f)
        df = pd.read_csv(osp.join(ddir, f), dtype={'instruction': str, 'response': str, 'cleaned_instruction': str, 'cleaned_response': str},keep_default_na=False,na_values=[])
        df['mol_list'] = df['mol_list'].apply(ast.literal_eval)
        print(len(df))
        #df[['mol_list', 'cleaned_instruction', 'cleaned_response']] = df.progress_apply(lambda x: pd.Series(extract_mol_content2(x['instruction'], x['response'])), axis=1)
        train_data.append((df, f.replace('.csv', ''),None))

        ddir = osp.join(self.instruction_data_path)
        f = 'SMolInstruct_val.csv'
        print(f)
        df = pd.read_csv(osp.join(ddir, f), dtype={'instruction': str, 'response': str, 'cleaned_instruction': str, 'cleaned_response': str},keep_default_na=False,na_values=[])
        df['mol_list'] = df['mol_list'].apply(ast.literal_eval)
        print(len(df))
        #df[['mol_list', 'cleaned_instruction', 'cleaned_response']] = df.progress_apply(lambda x: pd.Series(extract_mol_content2(x['instruction'], x['response'])), axis=1)
        valid_data.append((df, f.replace('.csv', ''),None))

        ddir = osp.join(self.instruction_data_path)
        f = 'SMolInstruct_test.csv'
        print(f)
        df = pd.read_csv(osp.join(ddir, f), dtype={'instruction': str, 'response': str, 'cleaned_instruction': str, 'cleaned_response': str},keep_default_na=False,na_values=[])
        df['mol_list'] = df['mol_list'].apply(ast.literal_eval)
        print(len(df))
        #df[['mol_list', 'cleaned_instruction', 'cleaned_response']] = df.progress_apply(lambda x: pd.Series(extract_mol_content2(x['instruction'], x['response'])), axis=1)
        test_data.append((df, f.replace('.csv', ''),None))


        print('Dataframes Loaded')

        # FIXME: test only
        #train_data = pd.read_csv(self.data_path + 'kinase_test.csv')
        #valid_data = pd.read_csv(self.data_path + 'kinase_test.csv')

        #print('length:', len(self.molecule_tokenizer.bad_blocks))
        #print(len(self.molecule_tokenizer))

        print('Molecule Building Block Input Created / Loaded')

        self.train_dses = []
        self.valid_dses = []
        self.test_dses = []

        for df, task, shrink in to_split_data:
            if task.startswith('tulu'): 
                ds_type = TuluDataset
            elif task.startswith('mol-inst'): 
                ds_type = MolInstDataset
            else: 
                ds_type = GeneralDataset

            ts = min(40, max(int(0.01*len(df)), 10), int(0.1*len(df)))
            if ts>1:
                train_df, val_df = train_test_split(df, test_size=ts, random_state = self.seed)
                val_df, test_df = train_test_split(val_df, test_size=0.5, random_state = self.seed)
            ds = ds_type(
                train_df,
                self.tokenizer,
                None,
                task_name=task,
                trunc_length=self.trunc_length,
                shrink_size = shrink,
            )
            self.train_dses.append(ds)
            if ts>0 and len(val_df) > 0:
                ds = ds_type(
                    val_df,
                    self.tokenizer,
                    None,
                    task_name=task,
                    trunc_length=self.trunc_length,
                )
                self.valid_dses.append(ds)
            if ts>0 and len(test_df) > 0:
                ds = ds_type(
                    test_df,
                    self.tokenizer,
                    None,
                    task_name=task,
                    trunc_length=self.trunc_length,
                )
                self.test_dses.append(ds)


        for df, task, shrink in train_data:
            if task.startswith('tulu'): ds_type = TuluDataset
            elif task.startswith('mol-inst'): ds_type = MolInstDataset
            else: ds_type = GeneralDataset
            ds = ds_type(
                df,
                self.tokenizer,
                None,
                task_name=task,
                trunc_length=self.trunc_length,
                shrink_size = shrink,
            )
            self.train_dses.append(ds)


        print('\n\n\nIndividual Dataset Lengths:')
        for ds in self.train_dses:
            print(ds.task_name, len(ds))


        self.train_ds = ConcatDataset(self.train_dses)
        self.train_ds = StatefulShuffleDataset(self.train_ds, seed=0)
        print('Total Training Data Length:', len(self.train_ds))
        #zz

        for df, task, shrink in valid_data:
            if task.startswith('tulu'): ds_type = TuluDataset
            elif task.startswith('mol-inst'): ds_type = MolInstDataset
            else: ds_type = GeneralDataset
            ds = ds_type(
                df,
                self.tokenizer,
                None,
                task_name=task,
                trunc_length=self.trunc_length,
                shrink_size = shrink,
            )
            self.valid_dses.append(ds)

        for df, task, shrink in test_data:
            if task.startswith('tulu'): ds_type = TuluDataset
            elif task.startswith('mol-inst'): ds_type = MolInstDataset
            else: ds_type = GeneralDataset
            ds = ds_type(
                df,
                self.tokenizer,
                None,
                task_name=task,
                trunc_length=self.trunc_length,
                shrink_size = shrink,
            )
            self.test_dses.append(ds)


    def train_dataloader(self):
        return CustomDataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=8,
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







################### To HF ############################




from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, HfFolder
import os

def push_total_module_to_hf(module: TotalDataModule, repo_id: str, private: bool = True):
    """
    Convert TotalDataModule splits into Hugging Face Datasets and push to hub.
    
    Args:
        module (TotalDataModule): Your initialized and setup DataModule.
        repo_id (str): e.g. "username/TotalDataModule"
        private (bool): Whether the repo should be private.
    """
    # Ensure setup is called
    module.setup("fit")

    def _convert(ds):
        """Convert a PyTorch Dataset into HuggingFace Dataset."""
        records = []
        for i in range(len(ds)):
            example = ds.get(i)
            # Flatten structure for HF compatibility
            flat_example = {
                "task_id": example["task_id"],
                "raw_instruction": example["raw_instruction"],
                "raw_response": example["raw_response"],
                "mol_list": example["mol_list"],
            }
            records.append(flat_example)
        return Dataset.from_list(records)

    # Convert splits
    train_ds = _convert(module.train_ds)
    valid_dses = [_convert(ds) for ds in module.valid_dses]
    test_dses = [_convert(ds) for ds in module.test_dses]

    # Merge validation and test datasets if multiple
    valid_ds = valid_dses[0].concatenate(valid_dses[1:]) if len(valid_dses) > 1 else (valid_dses[0] if valid_dses else None)
    test_ds = test_dses[0].concatenate(test_dses[1:]) if len(test_dses) > 1 else (test_dses[0] if test_dses else None)

    # Build DatasetDict
    dataset_dict = DatasetDict({"train": train_ds})
    if valid_ds: dataset_dict["validation"] = valid_ds
    if test_ds: dataset_dict["test"] = test_ds

    # Push to Hugging Face
    #dataset_dict.push_to_hub(repo_id, private=private)

    print(f"✅ Successfully pushed to https://huggingface.co/datasets/{repo_id}")


if __name__ == '__main__':

    config = {
        "pretrained_tokenizer": "Qwen/Qwen2.5-3B",
        "seed": 42,
        "GNN_cache": None,
        'instruction_data_path':'/shared/nas/data/m1/shared-resource/MoleculeLanguage/mCLM/instruction/dataloader_processed_onlyblocks_top_500/',
        'synthetic_data_path':'/shared/nas/data/m1/shared-resource/MoleculeLanguage/mCLM/synthetic/dataloader_processed_onlyblocks_top_500/',
        'downsample_tulu': 0.1,
    }

    module = TotalDataModule(config, base_model="llama3", batch_size=2, instruction_data_path=config['instruction_data_path'], \
        synthetic_data_path=config['synthetic_data_path'],
    )

    push_total_module_to_hf(module, repo_id="language-plus-molecules/mCLM_Pretrain_1k", private=True)

