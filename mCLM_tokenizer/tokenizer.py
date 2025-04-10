

from rdkit.Chem import rdDepictor
from rdkit import Chem

import copy

""" Implementation of the BRICS algorithm from Degen et al. ChemMedChem *3* 1503-7 (2008)
"""
import re
import sys
import json

from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit import RDRandom as random
from rdkit.Chem import rdChemReactions as Reactions
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import rdkit

from tqdm import tqdm
tqdm.pandas()
from rdkit.Chem.Scaffolds import MurckoScaffold

def canonicalize(smi):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi))





def get_aromatic_scaffold(mol):
    # Generate Murcko scaffold
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)

    # Identify aromatic atoms and their direct connections
    non_aromatic_atoms = set()
    for ring in Chem.GetSymmSSSR(scaffold):
        for atom in ring:
            if not scaffold.GetAtomWithIdx(atom).GetIsAromatic():
                non_aromatic_atoms.add(atom)
                for neighbor in scaffold.GetAtomWithIdx(atom).GetNeighbors():
                    non_aromatic_atoms.add(neighbor.GetIdx())

    # Create new molecule with only aromatic systems and their connections
    new_mol = Chem.RWMol()
    atom_map = {}

    for atom in scaffold.GetAtoms():
        if atom.GetIdx() not in non_aromatic_atoms:
            new_idx = new_mol.AddAtom(Chem.Atom(atom.GetAtomicNum()))
            atom_map[atom.GetIdx()] = new_idx

    for bond in scaffold.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a1 in atom_map and a2 in atom_map:
            new_mol.AddBond(atom_map[a1], atom_map[a2], bond.GetBondType())

    return new_mol.GetMol()

def find_main_chain(mol):
    scaffold = get_aromatic_scaffold(mol)
    substructure_match = mol.GetSubstructMatch(scaffold)

    return scaffold, substructure_match

def sub_idx(mol, substructure):

    all_indices = set(range(mol.GetNumAtoms()))

    substructure_match = mol.GetSubstructMatch(substructure)
    substructure_indices = set(substructure_match)
    non_substructure_indices = list(all_indices - substructure_indices)

    edit_mol = Chem.EditableMol(Chem.Mol())
    for idx in non_substructure_indices:
        edit_mol.AddAtom(mol.GetAtomWithIdx(idx))
    for i, idx1 in enumerate(non_substructure_indices):
        for j, idx2 in enumerate(non_substructure_indices[i+1:]):
            if mol.GetBondBetweenAtoms(idx1, idx2):
                edit_mol.AddBond(i, i+j+1, mol.GetBondBetweenAtoms(idx1, idx2).GetBondType())
    fragment_mol = edit_mol.GetMol()

    fragments = Chem.GetMolFrags(fragment_mol)
    connected_groups = []
    for fragment in fragments:
        group = [non_substructure_indices[idx] for idx in fragment]
        connected_groups.append(group)

    return connected_groups

def atom_idx(mol):
    scaffold, main_idx = find_main_chain(mol)
    subs_idx = sub_idx(mol, scaffold)

    for sub in subs_idx:
        if len(sub) > 7:
            for item in sub:
                main_idx+=(item, )

    return main_idx

import copy

""" Implementation of the BRICS algorithm from Degen et al. ChemMedChem *3* 1503-7 (2008)
"""
import re
import sys

from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit import RDRandom as random
from rdkit.Chem import rdChemReactions as Reactions
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import rdkit
from collections import defaultdict

environs = {
    'L1':'[#7&X3,#8&X2;!d1]', # N, O for amide bond / ester bond
    'L2':'[#6;^2;!$([#6X3]=[OX1])]', # C(sp2)
    'L3':'[#6;^3;d{2-3}]', # C(sp3) primary/secondary alkyl
    'L5':'[#6X3](=[OX1])', # Carbonyl group [#6X3;!$([#6X3](=[OX1])(-[#7])(-[#7]))](=[OX1])]
    'L6':'[#7&X3,#8&X2;^2;!d1]', # N,O for C-N, C-O bond

    # Ring (RS >= 8)
    'L4':'[C;X3;r{8-}](=[OX1])', # cyclic carbonyl group
    'L7':'[N&X3,O;^2;r{8-}]', # N, O for cyclic amide / ester bond
    'L8':'[C;^2;r{8-};!$([CX3]=[OX1])]', # cyclic C(sp2)
    'L9':'[C;^3;r{8-};d{2-3}]', # cyclic C(sp3) primary/secondary alkyl
    'L10':'[N&X3,O;^2;r{8-}]', # N,O for cyclic C-N, C-O bond
}

reactionDefs = (
  # L1
  [
    ('1', '5', '-;!@'), # Noncyclic amide bond, ester bond
  ],

  # L2
  [
    ('2', '2', '-;!@'), # sp2-sp2
    ('2', '3', '-;!@'), # sp2-sp3 primary, secondary alkyl
    ('2', '6', '-;!@'), # sp2-N, O
  ],

  # L3, L4
  # None left

  #L5
  [
    ('4', '7', '-'), # cyclic amide bond, ester bond (RS>=8)
  ],

  #L6, L7
  # None left

  #L8
  [
    ('8', '8', '-'), # cyclic sp2-sp2
    ('8', '9', '-'), # cyclic sp2-sp3 primary/seconday
    ('8', '10', '-'), # cyclic sp2-N, O
  ],

)
smartsGps = copy.deepcopy(reactionDefs)
for gp in smartsGps:
  for j, defn in enumerate(gp):
    g1, g2, bnd = defn
    r1 = environs['L' + g1]
    r2 = environs['L' + g2]
    g1 = re.sub('[a-z,A-Z]', '', g1)
    g2 = re.sub('[a-z,A-Z]', '', g2)
    sma = '[$(%s):1]%s[$(%s):2]>>[%s*]-[*:1].[%s*]-[*:2]' % (r1, bnd, r2, g1, g2)
    gp[j] = sma

for gp in smartsGps:
  for defn in gp:
    try:
      t = Reactions.ReactionFromSmarts(defn)
      t.Initialize()
    except Exception:
      print(defn)
      raise

environMatchers = {}
for env, sma in environs.items():
  environMatchers[env] = Chem.MolFromSmarts(sma)

bondMatchers = []
for i, compats in enumerate(reactionDefs):
  tmp = []
  for i1, i2, bType in compats:
    e1 = environs['L%s' % i1]
    e2 = environs['L%s' % i2]
    patt = '[$(%s)]%s[$(%s)]' % (e1, bType, e2)
    patt = Chem.MolFromSmarts(patt)
    tmp.append((i1, i2, bType, patt))
  bondMatchers.append(tmp)

reactions = tuple([[Reactions.ReactionFromSmarts(y) for y in x] for x in smartsGps])
reverseReactions = []
for i, rxnSet in enumerate(smartsGps):
  for j, sma in enumerate(rxnSet):
    rs, ps = sma.split('>>')
    sma = '%s>>%s' % (ps, rs)
    rxn = Reactions.ReactionFromSmarts(sma)
    labels = re.findall(r'\[([0-9]+?)\*\]', ps)
    rxn._matchers = [Chem.MolFromSmiles('[%s*]' % x) for x in labels]
    reverseReactions.append(rxn)

def FindSynthBonds(mol, randomizeOrder=False, silent=True):
  """ returns the machine synthesizable bonds in a molecule  """
  letter = re.compile('[a-z,A-Z]')
  indices = list(range(len(bondMatchers)))
  bondsDone = set()
  if randomizeOrder:
    random.shuffle(indices, random=random.random)

  atom_path = atom_idx(mol)

  envMatches = {}
  for env, patt in environMatchers.items():
    envMatches[env] = mol.HasSubstructMatch(patt)
  for gpIdx in indices:
    if randomizeOrder:
      compats = bondMatchers[gpIdx][:]
      random.shuffle(compats, random=random.random)
    else:
      compats = bondMatchers[gpIdx]
    for i1, i2, bType, patt in compats:
      if not envMatches['L' + i1] or not envMatches['L' + i2]:
        continue
      matches = mol.GetSubstructMatches(patt)
      i1 = letter.sub('', i1)
      i2 = letter.sub('', i2)
      for match in matches:
        if match not in bondsDone and (match[1], match[0]) not in bondsDone and (match[1] in atom_path and match[0] in atom_path):
          bondsDone.add(match)
          match_gen = (match[0], match[1])
          yield (tuple(match_gen), (i1, i2))


def fragment_on_bond(mol, atom1, atom2, bondtp1, bondtp2, addLabel=True):
    # Function to count dummy atoms in a molecule
    def count_dummy_atoms(molecule):
        return sum(1 for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 0)

    bond = mol.GetBondBetweenAtoms(atom1, atom2)
    new_mol = Chem.FragmentOnBonds(mol, [bond.GetIdx()], addDummies=addLabel, dummyLabels=[(0,0)])
    fragments = Chem.GetMolFrags(new_mol, asMols=True)

    if any(count_dummy_atoms(frag) > 2 or (frag.GetNumAtoms()-count_dummy_atoms(frag)) < 5 for frag in fragments):
        return mol

    return new_mol

def fragment_on_bond_order(mol, atom1, atom2, bondtp1, bondtp2, label=0):
    # Function to count dummy atoms in a molecule
    def count_dummy_atoms(molecule):
        return sum(1 for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 0)

    bond = mol.GetBondBetweenAtoms(atom1, atom2)
    new_mol = Chem.FragmentOnBonds(mol, [bond.GetIdx()], addDummies=True, dummyLabels=[(label, label)])
    fragments = Chem.GetMolFrags(new_mol, asMols=True)

    if any(count_dummy_atoms(frag) > 2 or (frag.GetNumAtoms()-count_dummy_atoms(frag)) < 5 for frag in fragments):
        return mol

    return new_mol


def get_fragment_size(mol, atom1, atom2):
    bond = mol.GetBondBetweenAtoms(atom1, atom2)
    new_mol = Chem.FragmentOnBonds(mol, [bond.GetIdx()], addDummies=True, dummyLabels=[(0,0)])
    fragments = Chem.GetMolFrags(new_mol)
    return min(len(frag) for frag in fragments)

def ReduceSynthBonds(bond_list, mol, ring=True, fragment_sizes=None):
    if fragment_sizes is None:
        fragment_sizes = {}
    indices = []
    for i in range(len(bond_list)):
        if ring == False and int(bond_list[i][1][0]) > 3:
            indices.append(i)
            continue
        for j in range(i+1, len(bond_list)):
            tuple1 = bond_list[i][0]
            tuple2 = bond_list[j][0]
            if any(element in tuple2 for element in tuple1):
                if (i not in indices) and (j not in indices):
                    # Get fragment sizes for both bonds
                    size_i = get_fragment_size(mol, tuple1[0], tuple1[1])
                    size_j = get_fragment_size(mol, tuple2[0], tuple2[1])

                    # Compare fragment sizes first, then bond priorities
                    if size_i > size_j:
                        indices.append(j)
                    elif size_i < size_j:
                        indices.append(i)
                    else:
                        # If sizes are equal, use bond priorities
                        if int(bond_list[i][1][0]) <= int(bond_list[j][1][0]):
                            indices.append(j)
                        else:
                            indices.append(i)

    for k in sorted(indices, reverse=True):
        del bond_list[k]
    return bond_list


def BreakSynthBonds(mol, reduce=True, ring=True, addLabels=True):
# reduce: pick one from crashing priorities / ring: include ring or not / addLabels: whether to add labels - possibly useful for enumerating module
    bonds_list = []
    if reduce==True:
        if ring==False:
            bond_list = ReduceSynthBonds(list(FindSynthBonds(mol)), mol, ring=False)
        else:
            bond_list = ReduceSynthBonds(list(FindSynthBonds(mol)), mol)
    else:
        bond_list = list(FindSynthBonds(mol))

    for i,bond in enumerate(bond_list):
        #mol=fragment_on_bond_order(mol, bond[0][0], bond[0][1], int(bond[1][0]), int(bond[1][1]), addLabel=addLabels)
        mol=fragment_on_bond_order(mol, bond[0][0], bond[0][1], int(bond[1][0]), int(bond[1][1]), label=i+2)

    return mol



def extract_numbers(smiles):
    matches = re.findall(r'\[(\d+)\*\]', smiles)
    if matches:
        return tuple(map(int, matches))
    return None

def remove_number_from_tuple(tup, num_to_remove):
    result = tuple(x for x in tup if x != num_to_remove)
    if len(result) == 1:
        return result[0]  # Return the single remaining element
    elif len(result) > 1:
        return result     # Return tuple if multiple elements remain
    else:
        return None       # Return None if the tuple is empty after removal


def reorder_fragments(frags):

    if frags == '': return ""

    frags = frags.split('.')

    #print(frags)

    fcount = [f.count('*]') for f in frags]
    fmax = max(fcount)
    fmin = min(fcount)
    if fmax > 2: return ""
    if fmin > 1: return ""
    if fmax == 0 and fmin == 0: return frags[0]

    new_frags = []
    for f in frags:
        if f.count('*]') == 1:
            new_frags.append(f)
            break
    frags.remove(f)
    #print(f, frags)

    index = extract_numbers(new_frags[0])[0]
    while len(frags) > 1:
        for f in frags:
            if f'[{index}*]' in f and f.count('*]') >= 2:
                new_nums = extract_numbers(f)
                index = remove_number_from_tuple(new_nums, index)
                new_frags.append(f)
                break
        frags.remove(new_frags[-1])
    new_frags.append(frags[-1])

    #new_frags = [nf.replace(f'{i}*', '0*').replace(f'{i+1}*', '1*') for i, nf in enumerate(new_frags)]
    new_frags2 = []
    old_index = -1
    index = extract_numbers(new_frags[0])[0]
    for i in range(len(new_frags)-1):
        nf = new_frags[i]
        nf2 = new_frags[i+1]
        new_frags2.append(nf.replace(f'[{index}*]', '[1*]').replace(f'[{old_index}*]', '[2*]'))

        old_index = index
        index = remove_number_from_tuple(extract_numbers(nf2), old_index)

    #print(index, old_index)
    new_frags2.append(new_frags[-1].replace(f'[{old_index}*]', '[2*]'))

    #print('.'.join(new_frags))
    #print('.'.join(new_frags2))

    return '^'.join(new_frags2)


def reverse_rfrags(frags):

    frags = frags.split('.')[::-1]

    return '^'.join(frags).replace('1*', '3*').replace('2*', '1*').replace('3*', '2*')


def get_blocks(smi, preprocessed=None):

    #csmi = canonicalize(smi)

    #if it's a reaction, process differently
    if '>' in smi:
        rfrags = [get_blocks(s, preprocessed=preprocessed)[0] for s in smi.split('>')]
        rrfrags = [get_blocks(s, preprocessed=preprocessed)[1] for s in smi.split('>')]
        if '' in rfrags:
            rfrags = ''
            rrfrags = ''
        else:
            rfrags = '>'.join(rfrags)
            rrfrags = '>'.join(rrfrags)
        if preprocessed:
            preprocessed[smi] = rfrags
            preprocessed['r_'+smi] = rrfrags
        return rfrags, rrfrags
    #process multiple molecules separately
    if '.' in smi:
        rfrags = [get_blocks(s, preprocessed=preprocessed)[0] for s in smi.split('.')]
        rrfrags = [get_blocks(s, preprocessed=preprocessed)[1] for s in smi.split('.')]
        if '' in rfrags:
            rfrags = ''
            rrfrags = ''
        else:
            rfrags = '.'.join(rfrags)
            rrfrags = '.'.join(rrfrags)
        if preprocessed:
            preprocessed[smi] = rfrags
            preprocessed['r_'+smi] = rrfrags
        return rfrags, rrfrags

    if preprocessed:
        if smi in preprocessed: return preprocessed[smi]

    s_mol = Chem.MolFromSmiles(smi)

    if s_mol == None: return ""

    #check for "*" in OPV molecules
    #star_flag = any([a.GetAtomicNum() == 0 and a.GetAtomMapNum() == 0 for a in s_mol.GetAtoms()])

    for a in s_mol.GetAtoms():
        if a.GetAtomicNum() == 0 and a.GetAtomMapNum() == 0:
            a.SetAtomicNum(1)
            a.SetBoolProp('star_flag', True)
    Chem.SanitizeMol(s_mol)
    #print(Chem.MolToSmiles(s_mol))

    frag_mol = BreakSynthBonds(s_mol, reduce=True, ring=True, addLabels=True)
    #print(Chem.MolToSmiles(frag_mol))
    for a in frag_mol.GetAtoms():
        if a.HasProp('star_flag') and a.GetBoolProp('star_flag'):
            a.SetAtomicNum(0)
    #print(Chem.MolToSmiles(frag_mol))
    Chem.SanitizeMol(frag_mol)
    #print(Chem.MolToSmiles(frag_mol))

    frags = Chem.MolToSmiles(frag_mol)

    rfrags = reorder_fragments(frags)
    rrfrags = reverse_rfrags(rfrags)

    if preprocessed:
        preprocessed[smi] = rfrags
        preprocessed['r_'+smi] = rrfrags

    return rfrags, rrfrags


if __name__ == '__main__':

    import pandas as pd

    outpath = '/shared/nas/data/m1/shared-resource/MoleculeLanguage/data/blockified/'
    file = 'train_input.txt'

    df = pd.read_csv(f"/shared/nas/data/m1/shared-resource/MoleculeLanguage/data/{file}", delimiter='\t', header=None)
    df.columns = ['name', 'description']

    kinases = pd.read_csv("/shared/nas/data/m1/shared-resource/MoleculeLanguage/data/kinase_inhibitor_list.txt", delimiter='\t')
    kinases.set_index('name', inplace=True)

    preprocessed = {}

    #for _, d in df.iterrows():
    def get_blocks(name):
        #smi = d['SMILES']
        #desc = df['description']
        #print(name)
        smi = kinases.loc[name]['smiles']

        if smi in preprocessed: return preprocessed[smi]

        s_mol = Chem.MolFromSmiles(smi)

        if s_mol == None: return ""

        frag_mol = BreakSynthBonds(s_mol, reduce=True, ring=True, addLabels=True)
        frags = Chem.MolToSmiles(frag_mol)

        rfrags = reorder_fragments(frags)

        preprocessed[smi] = rfrags

        preprocessed['r_'+smi] = reverse_rfrags(rfrags)

        return rfrags

    #blocks = get_blocks('imatinib')
    #rblocks = reorder_fragments(blocks)
    #zz

    df['blocks'] = df['name'].progress_apply(get_blocks)
    df['rblocks'] = df['blocks'].progress_apply(reverse_rfrags)

    print(df)

    df.to_csv(f"{outpath}{file}")

    #zz

    out = []
    for s in preprocessed:
        for s2 in preprocessed[s].split('.'):
          out.append(s2.replace('2*', '*').replace('1*', '*'))

    rev = {}
    for s in preprocessed:
        for s2 in preprocessed[s].split('.'):
            s2 = s2.replace('2*', '*').replace('1*', '*')
            s = s.replace('r_', '')
            if s2 in rev:
                rev[s2].add(s)
            else:
                rev[s2] = set([s])
    for k in rev: rev[k] = list(rev[k])

    counts = dict()
    for b in out:
        counts[b] = counts.get(b, 0) + 1

    for b in counts: counts[b] = int(counts[b]/2)

    counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    with open(outpath+'kinase_counts.txt', 'w') as convert_file:
        convert_file.write(json.dumps(counts, indent=4))

    with open(outpath+'kinase_tokens.txt', 'w') as convert_file:
        convert_file.write(json.dumps(preprocessed, indent=4))

    with open(outpath+'kinase_tokens_rev.txt', 'w') as convert_file:
        convert_file.write(json.dumps(rev, indent=4))


def convert_SMILES_strings(text):    
    pattern = re.compile(r'<SMILES>(.*?)<\/SMILES>', re.DOTALL)
    mol_list = pattern.findall(text)  # Extract SMILES content
    mol_list = [m.strip() for m in mol_list]

    mol_list = [get_blocks(m)[0] for m in mol_list]
    mol_list2 = copy.copy(mol_list)

    def replacer(match):
        if mol_list2:
            return f'[MOL] {mol_list2.pop(0)} [/MOL]'
        return match.group(0)  # Fallback in case something goes wrong

    restored_text = re.sub(r'<SMILES>(.*?)<\/SMILES>', replacer, text, count=len(mol_list2))

    return mol_list, restored_text.strip()


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
    fragments = [Chem.MolFromSmiles(frag) for frag in fragment_string.split('^')]
    
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

