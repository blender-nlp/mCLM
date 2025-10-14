
import pickle
import numpy as np


from rdkit import Chem
from sascorer import calculateScore  # Ensure sascorer.py is available

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def compute_sa_scores(smiles_list):
    scores = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            scores.append(np.nan)  # Invalid SMILES
        else:
            score = calculateScore(mol)
            if score == None: score = np.nan
            scores.append(score)
    return scores

def percent_valid_smiles(smiles_list):
    if not smiles_list:
        return 0.0  # Avoid division by zero

    smiles_list = [smi if smi != "" else "-1" for smi in smiles_list]
    valid_count = sum(1 for smi in smiles_list if Chem.MolFromSmiles(smi) is not None)
    percent_valid = 100.0 * valid_count / len(smiles_list)
    return percent_valid


def get_valid(smiles_list):
    smiles_list = [smi if smi != "" else "-1" for smi in smiles_list] #replace "" with a failed SMILES
    return [smi for smi in smiles_list if Chem.MolFromSmiles(smi) is not None]
    
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity

def tanimoto_similarities(smiles_list1, smiles_list2, radius=2, n_bits=2048):
    """
    Computes Tanimoto similarities between pairs of SMILES in two lists.

    Parameters:
        smiles_list1 (list of str): First list of SMILES strings.
        smiles_list2 (list of str): Second list of SMILES strings.
        radius (int): Radius for Morgan fingerprint (default=2).
        n_bits (int): Number of bits in the fingerprint (default=2048).

    Returns:
        List of similarity scores (float), or None if either SMILES is invalid.
    """
    similarities = []
    for smi1, smi2 in zip(smiles_list1, smiles_list2):
        mol1 = Chem.MolFromSmiles(smi1)
        mol2 = Chem.MolFromSmiles(smi2)
        if mol1 is None or mol2 is None:
            similarities.append(np.nan)
            continue
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius, nBits=n_bits)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius, nBits=n_bits)
        sim = TanimotoSimilarity(fp1, fp2)
        similarities.append(sim)
    return similarities





import os, sys
import torch

# Add the src directory to sys.path
sys.path.append(os.path.abspath('../../admet_oracle_model/src'))

from admet_oracle_model.src.main import prepare_dataset, evaluate, MLP

device = 'cpu'

def oracle_score(data_path, task):
    ckpt_path = '../../admet_oracle_model/checkpoints'

    task_ckpt_path = os.path.join(ckpt_path, f'{task}_mlp.pt')

    dataloader, smiles = prepare_dataset(data_path, ckpt_path)
    model = MLP().to(device)

    model.load_state_dict(torch.load(task_ckpt_path))
    all_preds = evaluate(model, dataloader, device).squeeze()

    return all_preds



def remove_dups(fda_smiles, smiles):
    all_done = set()
    new_smiles = []
    for fsmi, smi in zip(fda_smiles, smiles):
        if fsmi in all_done: 
            continue
        all_done.add(fsmi)
        new_smiles.append(smi)
    return new_smiles



def fix_claude(inp):
    fr = inp['full_response']

    rv = fr.split('<SMILES>')[1].strip()

    #print(rv)
    return rv

base = 'FineMolTex'

pth = '../../baseline/FineMolTex/scripts/FDA_edits_{}.FineMolTex.pkl'

for task in ['ames', 'bbbp', 'cyp3a4', 'dili', 'hia', 'pgp']:

    print('\t', task)

    with open(pth.format(task), 'rb') as f:
        data_to_save = pickle.load(f)

    #print(data_to_save.keys())



    mCLM_smiles, FDA_smiles, base_smiles = data_to_save['mCLM_smiles'], data_to_save['FDA_smiles'], data_to_save['MolSTM_smiles']

    #mCLM_smiles, FDA_smiles, MolSTM_smiles = mCLM_smiles[:10], FDA_smiles[:10], MolSTM_smiles[:10]
    #print(len(FDA_smiles), len(mCLM_smiles), len(base_smiles))

    base_smiles = remove_dups(FDA_smiles, base_smiles)
    FDA_smiles = remove_dups(FDA_smiles, FDA_smiles)
    #print(len(FDA_smiles), len(mCLM_smiles), len(base_smiles))
    

    FDA_SAs = compute_sa_scores(FDA_smiles)
    mCLM_SAs = compute_sa_scores(mCLM_smiles)
    MolSTM_SAs = compute_sa_scores(base_smiles)

    print('\t', 'FDA SAs:', np.nanmean(FDA_SAs), 'from {} percent valid'.format(percent_valid_smiles(FDA_smiles)))
    print('\t', 'mCLM SAs:', np.nanmean(mCLM_SAs), 'from {} percent valid'.format(percent_valid_smiles(mCLM_smiles)))
    print('\t', '{} SAs:'.format(base), np.nanmean(MolSTM_SAs), 'from {} percent valid'.format(percent_valid_smiles(base_smiles)))

    #print('\t', len(MolSTM_SAs))
    print()

    comb_scores = oracle_score(FDA_smiles + mCLM_smiles + get_valid(base_smiles), task)
    FDA_scores = comb_scores[:len(FDA_smiles)]
    mCLM_scores = comb_scores[len(FDA_smiles):2*len(FDA_smiles)]
    MolSTM_scores = comb_scores[2*len(FDA_smiles):]
    print('\t', len(comb_scores), len(FDA_scores), len(mCLM_scores), len(MolSTM_scores))

    print('\t', 'FDA Scores:', np.nanmean(FDA_scores), 'from {} percent valid'.format(percent_valid_smiles(FDA_smiles)))
    print('\t', 'mCLM Scores:', np.nanmean(mCLM_scores), 'from {} percent valid'.format(percent_valid_smiles(mCLM_smiles)))
    print('\t', '{} Scores:'.format(base), np.nanmean(MolSTM_scores), 'from {} percent valid'.format(percent_valid_smiles(base_smiles)))

    print()

    print('\t', 'mCLM Similarity:', np.nanmean(tanimoto_similarities(FDA_smiles, mCLM_smiles)))
    print('\t', '{} Similarity:'.format(base), np.nanmean(tanimoto_similarities(FDA_smiles, base_smiles)))

    print()
    print()





