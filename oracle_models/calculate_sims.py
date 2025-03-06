import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np

from tqdm import tqdm


# Load the dataset
df = pd.read_csv("data.csv")#.head(10000)

# Convert SMILES to RDKit molecules
df["rdkit_mol"] = df["molecule"].apply(Chem.MolFromSmiles)

# Compute Morgan fingerprints
radius = 2
n_bits = 2048
df["fingerprints"] = df["rdkit_mol"].apply(lambda mol: AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits) if mol else None)

# Compute Tanimoto similarity matrix
num_molecules = len(df)


fp_vecs = np.stack([np.array(a) for a in df["fingerprints"]])

if True:
    def tanimoto_similarity(a, b):
        intersection = np.logical_and(a, b).sum(axis=1).astype(float)
        union = np.logical_or(a, b).sum(axis=1).astype(float)
        #if union == 0:
        #    return 0.0
        #return intersection / union
        return np.divide(intersection, union, out=np.zeros_like(intersection), where=union!=0)

    similarity_matrix = np.zeros((num_molecules, num_molecules))
    

    for i in tqdm(range(num_molecules)):
        #for j in range(i, num_molecules):
        #    if df["fingerprints"].iloc[i] is not None and df["fingerprints"].iloc[j] is not None:
                #sim = DataStructs.FingerprintSimilarity(df["fingerprints"].iloc[i], df["fingerprints"].iloc[j])
        sim = tanimoto_similarity(fp_vecs[i,:], fp_vecs)
        similarity_matrix[i, :] = sim
        #similarity_matrix[:, i] = sim  # Symmetric matrix

def tanimoto_similarity_vectorized(fp_vecs):
    intersection = np.logical_and(fp_vecs[:, None, :], fp_vecs[None, :, :]).sum(axis=2)
    union = np.logical_or(fp_vecs[:, None, :], fp_vecs[None, :, :]).sum(axis=2)
    return np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union!=0)


def tanimoto_similarity_chunked(fp_vecs, chunk_size=100):
    num_molecules = fp_vecs.shape[0]
    similarity_matrix = np.zeros((num_molecules, num_molecules), dtype=float)

    for i in tqdm(range(0, num_molecules, chunk_size), desc="Computing similarity"):
        end = min(i + chunk_size, num_molecules)
        intersection = np.logical_and(fp_vecs[i:end, None, :], fp_vecs[None, :, :]).sum(axis=2)
        union = np.logical_or(fp_vecs[i:end, None, :], fp_vecs[None, :, :]).sum(axis=2)
        similarity_matrix[i:end, :] = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union != 0)

    return similarity_matrix

#similarity_matrix2 = tanimoto_similarity_chunked(fp_vecs, chunk_size=100)  # Adjust chunk_size as needed

#print((similarity_matrix == similarity_matrix2).all())

zz

# Convert to DataFrame and save
similarity_df = pd.DataFrame(similarity_matrix, index=df["molecule"], columns=df["molecule"])
similarity_df.to_parquet("tanimoto_similarity_matrix.parquet", compression="snappy")






