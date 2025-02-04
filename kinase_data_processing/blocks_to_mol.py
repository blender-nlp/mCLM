from rdkit import Chem
from rdkit.Chem import AllChem

import re

patt = re.compile("\[[0-9]*\*\]") #need this for assert because bug in rdkit brics implementation

def get_adj_bond(mol, aidx):
    a = mol.GetAtomWithIdx(aidx)
    bonds = a.GetBonds()
    matches = []
    for b in bonds:
        idx = -1
        if b.GetBeginAtomIdx() != aidx:
            idx = b.GetBeginAtomIdx()
        else:
            idx = b.GetEndAtomIdx()
    
    return idx

def remove_placeholder_atoms(mol):
    
    # Identify dummy atoms (atomic number 0)
    atoms_to_remove = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    
    # Remove identified atoms
    mol = Chem.EditableMol(mol)
    for idx in reversed(atoms_to_remove):  # Reverse to avoid reindexing issues
        mol.RemoveAtom(idx)
    
    # Convert back to SMILES
    clean_mol = mol.GetMol()
    return clean_mol


def join_fragments(fragment_string: str) -> Chem.Mol:
    """
    Joins molecular fragments into a single molecule based on attachment points.
    
    Parameters:
        fragment_string (str): A dot-separated string of molecular fragments with attachment points.
    
    Returns:
        Chem.Mol: A single RDKit molecule object after joining fragments.
    """
    fragments = [Chem.MolFromSmiles(frag) for frag in fragment_string.split('.')]
    
    if None in fragments:
        raise ValueError("One or more fragments could not be parsed.")
    
    rw_mol = Chem.RWMol()
    
    # Track attachment points and sequentially bond fragments
    for i in range(len(fragments)):
        frag = Chem.RWMol(fragments[i])
        
        attachment_atoms_1 = [atom.GetIdx() for atom in frag.GetAtoms() if atom.GetSymbol() == '*' and atom.GetIsotope() == 1]
        attachment_atoms_2 = [atom.GetIdx() for atom in frag.GetAtoms() if atom.GetSymbol() == '*' and atom.GetIsotope() == 2]
        
        adj_1 = get_adj_bond(frag, attachment_atoms_1[0]) if attachment_atoms_1 else None
        adj_2 = get_adj_bond(frag, attachment_atoms_2[0]) if attachment_atoms_2 else None

        if adj_1:
            frag.GetAtomWithIdx(adj_1).SetProp('ind', f'1_{i}')
        if adj_2:
            frag.GetAtomWithIdx(adj_2).SetProp('ind', f'2_{i}')

        rw_mol.InsertMol(frag)

            
    for i in range(0, len(fragments)-1):
        
        attachment_atoms_1 = [atom.GetIdx() for atom in rw_mol.GetAtoms() if atom.HasProp("ind") and atom.GetProp('ind') == f'1_{i}']
        attachment_atoms_2 = [atom.GetIdx() for atom in rw_mol.GetAtoms() if atom.HasProp("ind") and atom.GetProp('ind') == f'2_{i+1}']
        

        rw_mol.AddBond(attachment_atoms_1[0], attachment_atoms_2[0], Chem.BondType.SINGLE)
        
    mol = remove_placeholder_atoms(rw_mol)
    
    return mol


if __name__ == '__main__':


    test_blocks = '[1*]Oc1ccnc2cc(OC)c(OC)cc12.[2*]c1ccc([1*])cc1.[2*]NC(=O)C1(C(=O)N[1*])CC1.[2*]c1ccc(F)cc1'


    mol = join_fragments(test_blocks)

    smi = Chem.MolToSmiles(mol)

    print(smi)

