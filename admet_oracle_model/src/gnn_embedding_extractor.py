import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, DataLoader
from torch import nn
from rdkit import Chem
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

from gnn import GNNOnlyModel
from config import task, ckpt_path, batch_size, device




# === SMILES to Graph Function ===
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_features = torch.tensor([[atom.GetAtomicNum()] for atom in mol.GetAtoms()], dtype=torch.float)

    edge_indices, edge_features = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond.GetBondTypeAsDouble()
        edge_indices += [[i, j], [j, i]]
        edge_features += [[bond_type], [bond_type]]

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)

    return Data(x=atom_features, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)


CONFIG_DICT = {
    "gnn": {
        "node_dim": 1,             # set to the dimensionality of your node features
        "edge_dim": 1,              # set to the dimensionality of your edge features
        "hidden_dim_graph": 128,    # hidden dimension in the graph encoder
        "num_mp_layers": 3,         # number of message passing layers
        "dropout": 0.1,
        "aggr": "mean",
        "jk": "cat",
    }
}

def gnn_embedding_extractor(SMILES_test, ckpt_path):

    ckpt_path = os.path.join(ckpt_path, f'{task}_genentech.pt')

    graphs = [smiles_to_graph(s) for s in SMILES_test]
    test_dataset = [g for g in graphs if g is not None]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    config = CONFIG_DICT['gnn']
    num_cls = 1
    model = GNNOnlyModel(config, num_cls).to(device)
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    
    X = []
    with torch.no_grad():
        for batch in test_loader:#tqdm(test_loader, desc='GNN feature extracting ...'):
            batch = batch.to(device)
            embeddings = model(batch).squeeze()
            try:
                X.extend(embeddings.cpu().detach().tolist())
            except:
                X.append(embeddings.cpu().detach().tolist())
    return X

