import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np

from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol, MurckoScaffoldSmiles

from tqdm import tqdm

tqdm.pandas()

# Load the dataset
df = pd.read_csv("data.csv")

# Convert SMILES to RDKit molecules
#df["rdkit_mol"] = df["molecule"].progress_apply(Chem.MolFromSmiles)

# Compute Morgan fingerprints
#df["scaffolds"] = df["rdkit_mol"].progress_apply(lambda mol: GetScaffoldForMol(mol) if mol else None)

#df["scaffolds_SMILES"] = df["scaffolds"].progress_apply(lambda mol: Chem.MolToSmiles(mol) if mol else None)

df['scaffolds'] = df['molecule'].progress_apply(MurckoScaffoldSmiles)

df.to_csv(f'data_scaffolds.csv', index=False)



