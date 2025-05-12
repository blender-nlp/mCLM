import os
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm, trange
import json

import warnings
warnings.filterwarnings("ignore")

from farm_helpers import farm_tokenization
from farm_embedding_extractor import farm_embedding_extractor
from chemberta_embedding_extractor import chemberta_embedding_extractor
from gnn_embedding_extractor import gnn_embedding_extractor

from config import task, data_path, save_path, ckpt_path, batch_size, thresholds, device
ckpt_path = os.path.join(ckpt_path, f'{task}_mlp.pt')





def prepare_dataset(data, ckpt_path, batch_size=batch_size, shuffle=False):
    if isinstance(data, list):
        SMILES_test = data
    else:
        with open(data) as f:
            SMILES_test = f.readlines()
        SMILES_test = [i.strip() for i in SMILES_test]

    FARM_SMILES = farm_tokenization(SMILES_test)
    farm_feature = farm_embedding_extractor(FARM_SMILES, ckpt_path)
    gnn_feature = gnn_embedding_extractor(SMILES_test, ckpt_path)
    chemberta_feature = chemberta_embedding_extractor(SMILES_test, ckpt_path)
    assert len(farm_feature) == len(gnn_feature) == len(chemberta_feature)
    
    data = []
    for i in range(len(farm_feature)):#trange(len(farm_feature), desc=f'Create ensemble data for {task}'):
        data.append(farm_feature[i] + gnn_feature[i] + chemberta_feature[i])

    data = torch.tensor(data, dtype=torch.float32)
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, SMILES_test

class MLP(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, dropout=0.2):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.model(x)

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_batch in dataloader:#tqdm(dataloader, desc=f'Ensemble inference for {task}'):
            X_batch = X_batch[0].to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu())
    all_preds = torch.cat(all_preds).numpy()
    return all_preds



if __name__ == "__main__":
    dataloader, smiles = prepare_dataset(data_path)
    model = MLP().to(device)

    model.load_state_dict(torch.load(ckpt_path))
    all_preds = evaluate(model, dataloader, device).squeeze()

    data = dict()
    for i in range(len(smiles)):
        th = thresholds[task]
        if all_preds[i] >= th: pred = 1
        else: pred = 0
        data[smiles[i]] = pred
    
    with open(save_path, 'w') as f:
        json.dump(data, f)
    

