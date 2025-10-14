
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
            scores.append(score)
    return scores

def percent_valid_smiles(smiles_list):
    if not smiles_list:
        return 0.0  # Avoid division by zero

    valid_count = sum(1 for smi in smiles_list if Chem.MolFromSmiles(smi) is not None)
    percent_valid = 100.0 * valid_count / len(smiles_list)
    return percent_valid


def get_valid(smiles_list):

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








pth = '../../baseline/MolSTM/MoleculeSTM/demos/FDA_edits_{}.pkl'

with open('MolSTM_smiles.txt', 'w') as fmstm, open('FDA_smiles.txt', 'w') as ffda, open('mCLM_smiles.txt', 'w') as fmclm:
    for task in ['ames', 'bbbp', 'cyp3a4', 'dili', 'hia', 'pgp']:

        print(task)

        with open(pth.format(task), 'rb') as f:
            data_to_save = pickle.load(f)

        #print(data_to_save.keys())



        mCLM_smiles, FDA_smiles, MolSTM_smiles = data_to_save['mCLM_smiles'], data_to_save['FDA_smiles'], data_to_save['MolSTM_smiles']

            
        all_done = set()
        mCLM_smiles2, FDA_smiles2, MolSTM_smiles2 = [], [], []
        for mCLM_smi, FDA_smi, MolSTM_smi in zip(mCLM_smiles, FDA_smiles, MolSTM_smiles):
            if FDA_smi in all_done: 
                continue
            all_done.add(FDA_smi)
            mCLM_smiles2.append(mCLM_smi)
            FDA_smiles2.append(FDA_smi)
            MolSTM_smiles2.append(MolSTM_smi)

        print('mCLM Valid:', percent_valid_smiles(mCLM_smiles2))
        print('FDA Valid:', percent_valid_smiles(FDA_smiles2))
        print('MolSTM Valid:', percent_valid_smiles(MolSTM_smiles2))


        for smi in MolSTM_smiles2:#get_valid(MolSTM_smiles2):
            fmstm.write(smi + '\t' + task +  '\n')

        for smi in get_valid(FDA_smiles2):
            ffda.write(smi + '\t' + task +  '\n')

        for smi in get_valid(mCLM_smiles2):
            fmclm.write(smi + '\t' + task +  '\n')


