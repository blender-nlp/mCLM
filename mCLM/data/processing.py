import logging
import numpy as np
import os
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
        + onek_encoding_unk(
            int(atom.GetHybridization()), ATOM_FEATURES["hybridization"]
        )
        + [1 if atom.GetIsAromatic() else 0]
        + [atom.GetMass() * 0.01]
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
    d = smiles_to_bigraph(
        smiles,
        node_featurizer=feat,
        edge_featurizer=dgllife.utils.CanonicalBondFeaturizer(
            bond_data_field="edge_attr"
        ),
    )
    n = Data(
        x=d.ndata["x"],
        edge_attr=d.edata["edge_attr"],
        edge_index=torch.stack(d.edges()).long(),
    )
    if y is not None:
        n.y = y
    if mol_features is not None:
        n.mol_features = mol_features
    return n


class PropTask(Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    REGRESSION = "regression"
    MULTI_CLASSIFICATION = "multi_classification"

    def get_metrics(self) -> List[str]:
        if self.value in ("binary_classification", "multi_classification"):
            return ["auc", "auprc", "acc"]
        elif self.value == "regression":
            return ["mse", "mae"]

    def get_default_metric(self) -> str:
        if self.value in ("binary_classification", "multi_classification"):
            return "auc"
        elif self.value == "regression":
            return "mse" @ staticmethod

    def validation_names(m) -> Union[str, List[str]]:
        if isinstance(m, str):
            return "val/" + m
        elif isinstance(m, Sequence):
            return ["val/" + i for i in m]

    @staticmethod
    def test_names(m) -> Union[str, List[str]]:
        if isinstance(m, str):
            return "test/" + m
        elif isinstance(m, Sequence):
            return ["test/" + i for i in m]


_logger = logging.getLogger(__name__)


class MolecularDataset(torch.utils.data.Dataset):
    """Dataset class to load molecules data."""

    def __init__(
        self,
        smiles_list: List[str],
        y_list: np.ndarray = None,
        include_smiles: bool = False,
        index_predetermined_file: str = None,
    ):
        """

        Args:
            smiles_list (List[str]): A list of smiles.
            y_list (np.ndarray, optional): A list of labels. Defaults to None.
            include_smiles (bool, optional): If add smiles information to molecular data object. Defaults to False.
            index_predetermined_file (str, optional): The path to the file that provides index information on predetermined data split. Defaults to None.
        """
        self.smiles = smiles_list
        self.ids = self.smiles  # for deepchem compatibility
        self.y = y_list
        self._mols = None

        if index_predetermined_file is not None:
            with open(index_predetermined_file, "rb") as rf:
                self.index_predetermined = pickle.load(rf)

        if self.y is not None:
            assert len(self.smiles) == len(self.y)

        self.precomputed = False
        self.data_list = None  # precomputed data

        self.include_smiles = include_smiles

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> Data:
        if self.precomputed:
            return self.data_list[idx]

        smile = str(self.smiles[idx]).strip()

        n = self.smiles_to_data(
            smile,
            (
                torch.tensor(self.y[idx]).unsqueeze(0).float()
                if self.y is not None
                else None
            ),
        )

        if self.include_smiles:
            n.smiles = self.smiles[idx]

        return n

    def precompute(self, parallelize: bool = False):
        _logger.info("Precomputing data...")
        if parallelize:
            self.data_list = process_map(
                self.__getitem__, range(len(self.smiles)), max_workers=8, chunksize=256
            )
        else:
            self.data_list = [
                self.__getitem__(i) for i in tqdm(range(len(self.smiles)))
            ]

        self.precomputed = True

    @property
    def mols(self):
        if self._mols is not None:
            return self._mols
        _logger.info("Computing RDKit molecules...")
        self._mols = smiles_list_to_mols(self.smiles, parallelize=True)
        return self._mols

    @staticmethod
    def smiles_to_data(s: str, y: torch.Tensor = None) -> Data:
        """Staticmethod to convert smile string to torch geometric data object.
        Args:
            s (str): Smile string.
            y (torch.Tensor, optional): Label. Defaults to None.

        Returns:
            torch_geometric.data.Data: torch_geometric data object.
        """
        return smiles_to_data(s, y=y, mol_features=None)

    @staticmethod
    def load_df_dataset(
        df: pd.DataFrame,
        smiles_column_name: str = "SMILES",
        y_column_names: List[str] = ("Activity",),
        index_predetermined_file: str = None,
    ) -> torch.utils.data.Dataset:
        """Staticmethod to load dataframe to MolecularDataset object.

        Args:
            df (pd.DataFrame): A dataframe that contains smiles column
            smiles_column_name (str, optional): Name of smiles column. Defaults to "SMILES".
            y_column_names (List[str], optional): List of label names. Defaults to ("Activity",).
            index_predetermined_file (str, optional): Path of file that provides predetermined split indexes. Defaults to None.

        Returns:
            torch.utils.data.Dataset: MolecularDataset object.
        """

        if len(y_column_names) == 1:
            # need to skip na for single task
            df = df.dropna(subset=y_column_names)

        dataset = df
        smiles = dataset[smiles_column_name].values

        if not y_column_names:
            y = None
        else:
            y = dataset[list(y_column_names)].values

        return MolecularDataset(
            smiles_list=smiles,
            y_list=y,
            index_predetermined_file=index_predetermined_file,
        )

    @staticmethod
    def load_csv_dataset(
        file_path: str,
        smiles_column_name: str = "SMILES",
        y_column_names: List[str] = ("Activity",),
        index_predetermined_file=None,
    ) -> torch.utils.data.Dataset:
        """Staticmethod to load csv file path to create MolecularDataset object.

        Args:
            df (pd.DataFrame): A dataframe that contains smiles column
            smiles_column_name (str, optional): Name of smiles column. Defaults to "SMILES".
            y_column_names (List[str], optional): List of label names. Defaults to ("Activity",).
            index_predetermined_file (str, optional): Path of file that provides predetermined split indexes. Defaults to None.

        Returns:
            torch.utils.data.Dataset: MolecularDataset object.
        """
        dataset = pd.read_csv(file_path)
        return MolecularDataset.load_df_dataset(
            dataset,
            smiles_column_name=smiles_column_name,
            y_column_names=y_column_names,
            index_predetermined_file=index_predetermined_file,
        )

    @staticmethod
    def load_ftr_dataset(
        file_path: str,
        smiles_column_name: str = "SMILES",
        y_column_names: List[str] = ("Activity",),
        index_predetermined_file=None,
    ) -> torch.utils.data.Dataset:
        """Staticmethod to load feather file path to create MolecularDataset object.

        Args:
            df (pd.DataFrame): A dataframe that contains smiles column
            smiles_column_name (str, optional): Name of smiles column. Defaults to "SMILES".
            y_column_names (List[str], optional): List of label names. Defaults to ("Activity",).
            index_predetermined_file (str, optional): Path of file that provides predetermined split indexes. Defaults to None.

        Returns:
            torch.utils.data.Dataset: MolecularDataset object.
        """
        dataset = pd.read_feather(file_path)
        return MolecularDataset.load_df_dataset(
            dataset,
            smiles_column_name=smiles_column_name,
            y_column_names=y_column_names,
            index_predetermined_file=index_predetermined_file,
        )

    @staticmethod
    def load_benchmark_dataset(
        name: str,
        root_path: str = "/projects/site/gred/resbioai/selettore/molecular-data",
    ) -> torch.utils.data.Dataset:
        if name == "hiv":
            return MolecularDataset.load_csv_dataset(
                os.path.join(root_path, "HIV.csv"),
                smiles_column_name="smiles",
                y_column_names=("HIV_active",),
            )
        elif name == "clintox":
            return MolecularDataset.load_csv_dataset(
                os.path.join(root_path, "clintox_clean.csv"),
                smiles_column_name="smiles",
                y_column_names=("CT_TOX",),
            )
        elif name == "tox21":
            return MolecularDataset.load_csv_dataset(
                os.path.join(root_path, "tox21_singleassay_clean.csv"),
                smiles_column_name="smiles",
                y_column_names=("NR-AR",),
            )
        elif name == "ames":
            return MolecularDataset.load_csv_dataset(
                os.path.join(root_path, "AMES_All_Data_preprocessed_clean.csv"),
                smiles_column_name="SMILES",
                y_column_names=("Activity",),
            )
        elif name == "lipo":
            return MolecularDataset.load_csv_dataset(
                os.path.join(root_path, "Lipophilicity.csv"),
                smiles_column_name="smiles",
                y_column_names=("exp",),
            )
        elif name == "freesolv":
            return MolecularDataset.load_csv_dataset(
                os.path.join(root_path, "freesolv_clean.csv"),
                smiles_column_name="smiles",
                y_column_names=("expt",),
            )
        elif name == "esol":
            return MolecularDataset.load_csv_dataset(
                os.path.join(root_path, "esol_clean.csv"),
                smiles_column_name="smiles",
                y_column_names=("measured log solubility in mols per litre",),
            )
        else:
            raise ValueError(f"Dataset not implemented for {name}")


class MolecularSubset(Subset):
    """A dataset class that create subset of a given MolecularDataset according to given indices."""

    def __init__(self, dataset: MolecularDataset, indices: np.ndarray):
        """
        Args:
            dataset (MolecularDataset): A MolecularDataset object.
            indices (np.ndarray): Indices for selection.
        """
        super().__init__(dataset, indices)
        self.smiles = (
            self.dataset.smiles[self.indices]
            if self.dataset.smiles is not None
            else None
        )
        self.y = self.dataset.y[self.indices] if self.dataset.y is not None else None

    def __getitem__(self, idx) -> Data:
        item = super().__getitem__(idx)
        return item

    def __len__(self) -> int:
        return len(self.smiles)

    def __getattr__(self, item) -> Data:
        return getattr(self.dataset, item)

    def recompute_y(self):
        self.y = self.dataset.y[self.indices] if self.dataset.y is not None else None

    def to_dataset(self) -> MolecularDataset:
        d = MolecularDataset(
            smiles_list=self.smiles, y_list=self.y, include_smiles=self.include_smiles
        )
        if hasattr(self.dataset, "mol_features"):
            d.mol_features = self.dataset.mol_features[self.indices, :]
        return d


def index_predetermined_split(
    dataset: MolecularDataset, seed: int = 0, remove_na: bool = False
):
    """Function that split dataset by predetermined index.

    Args:
        dataset (MolecularDataset): MolecularDataset object to be split.
        seed (int, optional): Seed to select from provided predetermined index sets. Defaults to 0.
        remove_na (bool, optional): If remove NA values after getting index_predetermined split index. Defaults to False.

    Returns:
        Tuple[MolecularDataset, MolecularDataset, MolecularDataset]: train/val/test dataset
    """
    split_indices = dataset.index_predetermined[seed]

    if len(split_indices) != 3:
        raise ValueError(
            "Split indices must have three splits: train, validation, and test"
        )

    train_ix, val_ix, test_ix = split_indices

    if (dataset.y.shape[1] == 1) and (remove_na == True):  # for single task
        _logger.info("Removing nan after splitting data by given predetermined index.")
        # get nan index
        nan_idx = np.argwhere(np.isnan(dataset.y.flatten())).flatten()
        # remove nan index
        train_ix = np.setdiff1d(train_ix, nan_idx)
        val_ix = np.setdiff1d(val_ix, nan_idx)
        test_ix = np.setdiff1d(test_ix, nan_idx)

    train = MolecularSubset(dataset, train_ix)
    val = MolecularSubset(dataset, val_ix)
    test = MolecularSubset(dataset, test_ix)
    return train, val, test
    