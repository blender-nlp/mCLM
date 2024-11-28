

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
  """ returns the machine synthesizable bonds in a molecule that BRICS would cleave    """
  letter = re.compile('[a-z,A-Z]')
  indices = list(range(len(bondMatchers)))
  bondsDone = set()
  if randomizeOrder:
    random.shuffle(indices, random=random.random)

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
        if match not in bondsDone and (match[1], match[0]) not in bondsDone:
          bondsDone.add(match)
          match_gen = (match[0], match[1])
          yield (tuple(match_gen), (i1, i2))

def fragment_on_bond(mol, atom1, atom2, bondtp1, bondtp2, addLabel=True):
    bond = mol.GetBondBetweenAtoms(atom1, atom2)
    new_mol = Chem.FragmentOnBonds(mol, [bond.GetIdx()], addDummies=addLabel, dummyLabels=[(bondtp1, bondtp2)])
    #new_mol = Chem.FragmentOnBonds(mol, [bond.GetIdx()], dummyLabels=[(bondtp1, bondtp2)])
    # FragmentOnBonds() calls ClearComputedProps() at the end.  There
    # is a current bug report where, as a downstream effect, that may
    # cause some chiralities to change, most notably on some
    # bridgeheads.. A workaround for now is to call SanitizeMol(),
    # though that ends up tripling the time. I'll stay compatible
    # with FragmentOnBonds() and not call it.
    Chem.SanitizeMol(new_mol)

    return new_mol



def fragment_on_bond_order(mol, atom1, atom2, bondtp1, bondtp2, label=0):
    bond = mol.GetBondBetweenAtoms(atom1, atom2)
    new_mol = Chem.FragmentOnBonds(mol, [bond.GetIdx()], addDummies=True, dummyLabels=[(label, label)])
    #new_mol = Chem.FragmentOnBonds(mol, [bond.GetIdx()], dummyLabels=[(bondtp1, bondtp2)])
    # FragmentOnBonds() calls ClearComputedProps() at the end.  There
    # is a current bug report where, as a downstream effect, that may
    # cause some chiralities to change, most notably on some
    # bridgeheads.. A workaround for now is to call SanitizeMol(),
    # though that ends up tripling the time. I'll stay compatible
    # with FragmentOnBonds() and not call it.
    Chem.SanitizeMol(new_mol)

    return new_mol

def ReduceSynthBonds(bond_list, ring=True): # pick one from crashing priorities / include ring or not
    indices = []
    for i in range(len(bond_list)):
        if ring==False and int(bond_list[i][1][0]) >= 4:
            indices.append(i)
            continue
        for j in range(len(bond_list)):
            tuple1 = bond_list[i][0]
            tuple2 = bond_list[j][0]
            if i!=j and any(element in tuple2 for element in tuple1):
                if (i not in indices) and (j not in indices):
                    if int(bond_list[i][1][0]) <= int(bond_list[j][1][0]):
                        indices.append(j)
                        continue
                    if int(bond_list[i][1][0]) >= int(bond_list[j][1][0]):
                        indices.append(i)

    for k in sorted(indices, reverse=True):
        del bond_list[k]
    return bond_list


def BreakSynthBonds(mol, reduce=True, ring=True, addLabels=True):
# reduce: pick one from crashing priorities / ring: include ring or not / addLabels: whether to add labels - possibly useful for enumerating module
    if reduce==True:
        if ring==False:
            bond_list = ReduceSynthBonds(list(FindSynthBonds(mol)), ring=False)
        else:
            bond_list = ReduceSynthBonds(list(FindSynthBonds(mol)))
    else:
        bond_list = list(FindSynthBonds(mol))

    for i,bond in enumerate(bond_list):
        #mol=fragment_on_bond(mol, bond[0][0], bond[0][1], int(bond[1][0]), int(bond[1][1]), addLabel=addLabels)
        mol=fragment_on_bond_order(mol, bond[0][0], bond[0][1], int(bond[1][0]), int(bond[1][1]), label=i+1)

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

    fcount = [f.count('*') for f in frags]
    fmax = max(fcount)
    fmin = min(fcount)
    if fmax > 2: return ""
    if fmin > 1: return ""
    if fmax == 0 and fmin == 0: return frags[0]

    new_frags = []
    for f in frags:
        if f.count('*') == 1:
            new_frags.append(f)
            break
    frags.remove(f)
    #print(f, frags)

    index = extract_numbers(new_frags[0])[0]
    while len(frags) > 1:
        for f in frags:
            if f'{index}*' in f and f.count('*') >= 2:
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
        new_frags2.append(nf.replace(f'{index}*', '0*').replace(f'{old_index}*', '1*'))

        old_index = index
        index = remove_number_from_tuple(extract_numbers(nf2), old_index)

    #print(index, old_index)
    new_frags2.append(new_frags[-1].replace(f'{old_index}*', '1*'))

    #print('.'.join(new_frags))
    #print('.'.join(new_frags2))

    return '.'.join(new_frags2)


def reverse_rfrags(frags):

    frags = frags.split('.')[::-1]

    return '.'.join(frags).replace('0*', '2*').replace('1*', '0*').replace('2*', '0*')



if __name__ == '__main__':

    import pandas as pd

    outpath = '/shared/nas/data/m1/shared-resource/MoleculeLanguage/data/blockified/'
    file = 'valid_input.txt'

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

    zz

    out = []
    for s in preprocessed:
        for s2 in preprocessed[s].split('.'):
          out.append(s2)

    counts = dict()
    for b in out:
            counts[b] = counts.get(b, 0) + 1

    counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    with open(outpath+'kinase_counts.txt', 'w') as convert_file: 
     convert_file.write(json.dumps(counts))

