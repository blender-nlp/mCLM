import os, sys
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm, trange
import json

import warnings
warnings.filterwarnings("ignore")

os.environ['TOKENIZERS_PARALLELISM'] = '0'

# Add the src directory to sys.path
sys.path.append(os.path.abspath('../admet_oracle_model/src'))


from farm_helpers import farm_tokenization
from farm_embedding_extractor import farm_embedding_extractor
from chemberta_embedding_extractor import chemberta_embedding_extractor
from gnn_embedding_extractor import gnn_embedding_extractor

from config import thresholds

from main import prepare_dataset, evaluate, MLP

tasks = ['ames', 'bbbp', 'cyp3a4', 'dili', 'hia', 'pgp']

device = 'cpu'
#data_path = 'fangels.txt'
ckpt_path = '../admet_oracle_model/checkpoints'

def score(data_path, task):
    task_ckpt_path = os.path.join(ckpt_path, f'{task}_mlp.pt')

    dataloader, smiles = prepare_dataset(data_path, ckpt_path)
    model = MLP().to(device)

    model.load_state_dict(torch.load(task_ckpt_path))
    all_preds = evaluate(model, dataloader, device).squeeze()

    data = dict()
    for i in range(len(smiles)):
        
        data[smiles[i]] = all_preds[i]

        continue

        th = thresholds[task]
        if all_preds[i] >= th: pred = 1
        else: pred = 0
        data[smiles[i]] = pred

    #with open(save_path, 'w') as f:
    #    json.dump(data, f)

    return data

if __name__ == '__main__':
    print('Fallen Angels Scores:')
    for task in tasks:
        print(task)

        scores = score('fangels.txt')

        print(scores)





