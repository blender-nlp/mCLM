

import json

import sys

import os

# Add the src directory to sys.path
sys.path.append(os.path.abspath('admet_oracle_model/src'))

from admet_oracle_model.src.main import prepare_dataset, evaluate, MLP


from rdkit import Chem

import subprocess

from mCLM.data.processing import insert_sublists, find_first_occurrence

import json

import torch

def canonicalize(smi):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def oracle_score(data_path, task):
    ckpt_path = 'admet_oracle_model/checkpoints'

    task_ckpt_path = os.path.join(ckpt_path, f'{task}_mlp.pt')

    dataloader, smiles = prepare_dataset(data_path, ckpt_path)
    model = MLP().to(device)

    model.load_state_dict(torch.load(task_ckpt_path))
    all_preds = evaluate(model, dataloader, device).squeeze()

    return all_preds


if len(sys.argv) < 2:
    print("Usage: python score_angels.py <filename>")
    sys.exit(1) # Exit with an error code

filename = sys.argv[1]



with open(filename) as f:
    data = json.load(f)



unique_smiles = set()


def recurse(data, prefix = '', scores=None, depth = 1):
    #print(data[0])
    

    if len(data) == 2 and isinstance(data[1], dict): 
        #return [recurse(data[1][d], prefix = str(data[0]) + f"--{d}-->") for d in data[1]]

        rvs = []
        for d in data[1]:
            if scores != None:
                s = scores[data[0][0]]
                pfx = prefix + str(data[0] + [f'Scores: {s}']) + f" ---{d}-->\n" + "    "*depth
            else:
                pfx = prefix + str(data[0]) + f" ---{d}-->\n" + "    "*depth

            res = recurse(data[1][d], prefix = pfx, scores=scores, depth=depth+1)
            for re in res:
                if scores is None: unique_smiles.add(data[0][0])
                rvs.append(re.strip())
        return rvs
        #print(rvs)
        #zz

    else:
        #print(prefix, data)
        if scores is None: unique_smiles.add(data[0])

        if scores != None:
            s = scores[data[0]]
            return [prefix + str(data + [f'Scores: {s}'])]
        else:
            return [prefix + str(data)]


    #print([data[1][d] for d in data[1]])
    #print()


for d in recurse(data):
    pass#
    #print(d)
    #print()

unique_smiles = list(unique_smiles) #Ensure order is preserved

props = ['ames', 'bbbp', 'cyp3a4', 'dili', 'hia', 'pgp']


def swap_keys(scores):
    swapped = {}

    for dataset, smiles_scores in scores.items():
        for smiles, val in smiles_scores.items():
            if smiles not in swapped:
                swapped[smiles] = {}
            swapped[smiles][dataset] = val
    return swapped


def get_scores(unique_smiles):
    rv = {}

    for p in props:
        scores = oracle_score(unique_smiles, p)

        rv[p] = {smi:round(s.item(),3) for smi, s in zip(unique_smiles, scores)}

    return swap_keys(rv)

scores = get_scores(unique_smiles)


for d in recurse(data, scores=scores):
    print(d)
    print()


print()

print(len(unique_smiles), 'Unique SMILES:', unique_smiles)
print()

print('Scores:', scores)














