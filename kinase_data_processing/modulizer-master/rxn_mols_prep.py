import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem


def sanitize(list):
    res = []
    for mol in list:
        mol_sanitized = Chem.MolToSmiles(Chem.MolFromSmiles(mol))
        res.append(mol_sanitized)
    return res


def sanitize_mol(mol):
    mol_sanitized = Chem.MolToSmiles(Chem.MolFromSmiles(mol))
    return mol_sanitized


def smilesListToMol(smiles_list):
    res= []
    for smiles in smiles_list:
        smiles_mod = Chem.MolFromSmiles(smiles)
        res.append(smiles_mod)
    return res
# draws all retrons
