import logging
import numpy as np
import os
import io
import pandas as pd
import torch
from torch_geometric.data import Data
from math import isclose
from random import Random
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from typing import Union, List, Tuple, Iterable
import pickle
from collections.abc import Sequence
from enum import Enum
from typing import Union, List
import dgllife
from dgllife.utils import smiles_to_bigraph

import re

from rdkit import Chem



MAX_ATOMIC_NUM = 100
ATOM_FEATURES = {
    "atomic_num": list(range(MAX_ATOMIC_NUM)),
    "degree": [0, 1, 2, 3, 4, 5],
    "formal_charge": [-1, -2, 1, 2, 0],
    "chiral_tag": [0, 1, 2, 3],
    "num_Hs": [0, 1, 2, 3, 4],
    "hybridization": [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
    "placeholders": [0, 1, 2, 3],
}


def atom_features(
    atom: Chem.rdchem.Atom, functional_groups: List[int] = None
) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """
    features = (
        onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES["atomic_num"])
        + onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES["degree"])
        + onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES["formal_charge"])
        + onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES["chiral_tag"])
        + onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES["num_Hs"])
        + onek_encoding_unk(int(atom.GetAtomMapNum()), ATOM_FEATURES["placeholders"])
        + onek_encoding_unk(
            int(atom.GetHybridization()), ATOM_FEATURES["hybridization"]
        )
        + [1 if atom.GetIsAromatic() else 0]
        + [atom.GetMass() * 0.01]
        #add a one hot encoding here based on the placeholder
    )  # scaled to about the same range as other features
    if functional_groups is not None:
        features += functional_groups
    return features


def atoms_features(mol):
    return {"x": torch.tensor([atom_features(atom) for atom in mol.GetAtoms()])}


def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding with an extra category for uncommon values.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
             If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def smiles_to_mol(s: str) -> Chem.Mol:
    return Chem.MolFromSmiles(s)


def smiles_list_to_mols(smiles: List[str], parallelize: bool = False) -> List[Chem.Mol]:
    if parallelize:
        mols = process_map(smiles_to_mol, smiles, max_workers=16, chunksize=1000)
    else:
        mols = [smiles_to_mol(s) for s in tqdm(smiles)]
    return mols

import re

def replace_star_notation(s):
    return re.sub(r'\[(\d+)\*\]', r'[*:\1]', s)

def smiles_to_data(
    smiles: str, y: torch.Tensor = None, mol_features: torch.Tensor = None
):
    """
    Featurizer for SMILES

    :param smiles:
    :param y:
    :param mol_features:
    :return:
    """
    feat = atoms_features

    smiles = replace_star_notation(smiles)

    edge_feat = dgllife.utils.CanonicalBondFeaturizer(
            bond_data_field="edge_attr"
        )
    edge_feat_dim = edge_feat.feat_size('edge_attr')
    #print(edge_feat_dim)
    #zz

    d = smiles_to_bigraph(
        smiles,
        node_featurizer=feat,
        edge_featurizer=edge_feat,
    )
    n = Data(
        x=d.ndata["x"],
        edge_attr=d.edata["edge_attr"] if "edge_attr" in d.edata else torch.zeros((0,edge_feat_dim)), #my modification to allow things like 'C'
        edge_index=torch.stack(d.edges()).long(),
    )
    if y is not None:
        n.y = y
    if mol_features is not None:
        n.mol_features = mol_features
    return n



def extract_mol_content(text):
    pattern = re.compile(r'\[MOL\](.*?)\[/MOL\]', re.DOTALL)
    mol_list = pattern.findall(text)  # Extract MOL contents
    mol_list = [m.strip() for m in mol_list]

    cleaned_text = pattern.sub('[MOL][/MOL]', text)  # Remove MOL content from text
    return mol_list, cleaned_text.strip()


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


def canonicalize(smi):
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    except:
        return None


def load_with_tqdm(file_path, map_location=None, weights_only=True):
    file_size = os.path.getsize(file_path)
    buffer_size = 1024 * 1024  # 1MB chunks
    
    with open(file_path, 'rb') as f, tqdm(desc='Loading', total=file_size, unit='iB', unit_scale=True, unit_divisor=1024) as pbar:
        buffer = bytearray()
        while True:
            chunk = f.read(buffer_size)
            if not chunk:
                break
            buffer.extend(chunk)
            pbar.update(len(chunk))
        
        byte_stream = io.BytesIO(buffer)
        data = torch.load(byte_stream, map_location=map_location, weights_only=weights_only)
    return data