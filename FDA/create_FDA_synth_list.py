
import pandas as pd
import ast

from rdkit import Chem

from tqdm import tqdm
tqdm.pandas()


seed = 42
df = pd.read_csv(f"drugs_dedupped.synth_modules.csv.forcoveragefull.csv", delimiter = ';')

#df = df.head(400)

print(len(df))

def get_bbs(smiles_pre):
    if type(smiles_pre['monoDeprotected']) == float:
        if type(smiles_pre['diDeprotected']) == float:
            sm = ''
        else:
            sm = smiles_pre['diDeprotected']
    elif type(smiles_pre['diDeprotected']) == float:
        sm = smiles_pre['monoDeprotected']
    else:
        sm = smiles_pre['monoDeprotected'] + "." + smiles_pre['diDeprotected']
    #smiles.append(sm)
    return sm

#print(df[df['total#ofBuilings']>0])
#df[df['total#ofBuilings']>0].to_csv('tmp.csv')
print('Sum of # BBs', df['total#ofBuilings'].sum())


df['BBs'] = df[['monoDeprotected', 'diDeprotected']].apply(get_bbs, axis=1)
print(len(df[df['BBs']!='']))
#df[df['BBs']!=''].to_csv('tmp.csv')
print('Got BBs')

#print(df['BBs'])
#zz
def without_isotopes(mol):
    if mol is None:
        return Chem.MolFromSmiles('')
    atom_data = [(atom, atom.GetIsotope()) for atom in mol.GetAtoms()]
    for atom, isotope in atom_data:
      if atom.GetAtomicNum() == 7 and isotope == 1:
          atom.SetIsotope(0)
    out = mol
    return out

df['target'] = df['target'].apply(Chem.MolFromSmiles).apply(without_isotopes).apply(Chem.MolToSmiles)
print('Target isotopes removed')

print(len(df))

from forward_synth import forward_synthesis, reaction_rules, deprotection_rules, convert_rules, run_with_timeout


def tokenize(row):

    #print(row)
    #zz

    smi = row['target']

    #Sara's algorithm was applied to separate components
    if '.' in smi:
        rv = [tokenize(s) for s in smi.split('.')]
        #print(rv)
        #print('.'.join([r[0] for r in rv]), ';'.join([r[1] for r in rv]))
        #zz
        #algos = ';'.join([r[1] for r in rv])
        blocks = [r[0] for r in rv]
        if not '' in blocks:
            blocks = '.'.join(blocks)
            return blocks
        else: return None

    target = smi
    building_blocks = row['BBs'].split('.')
    
    if building_blocks == ['']: return ''
    
    #zz
    #fs = forward_synthesis(target, building_blocks, reaction_rules, deprotection_rules, convert_rules)
    try:
        fs = run_with_timeout(60, forward_synthesis, target, building_blocks, reaction_rules, deprotection_rules, convert_rules)
    except TimeoutError as e:
        return ''

    if len(fs) == 0:
        return ''#raise
    if fs[0].startswith('^') or fs[0].endswith('^'): #there's a bug that does this in two molecules, so we'll throw those out
        # "[H]/N=C(/NOC(=O)OCC)c1ccc(-c2cnc(N3CCOCC3)c3nc(C=Cc4ccc5ccccc5n4)cn23)cn1" -> "^[1*]C=Cc1cn2c([2*])cnc(N3CCOCC3)c2n1^[2*]c1ccc2ccccc2n1"
        raise
    return fs[0]



df['blocks'] = df.progress_apply(tokenize, axis=1)

#print('Have blocks:', df['blocks'].apply(lambda x: len(x) > 0).sum() )


df = df[df['blocks'].apply(lambda x : len(x)>0)]

print('Have block:', len(df))


write_df = df[['target', 'blocks']].copy()
write_df = write_df.rename(columns={'target': 'SMILES'})

write_df.to_csv('drugs_dedupped.synth.csv')

df = df[df['blocks'].apply(lambda x : len(x.split('^'))>1)]

print('Have > 1 blocks', len(df))

df = df[df['blocks'].apply(lambda x : len(x.split('^'))>2)]
#df = df[df['blocks'].apply(lambda x : max([len(a.split('^')) for a in x[0].split('.')])>2)]

print('Have > 2 blocks', len(df))

df = df[df['blocks'].apply(lambda x : len(x.split('^'))>3)]
#df = df[df['blocks'].apply(lambda x : max([len(a.split('^')) for a in x[0].split('.')])>2)]

print('Have > 3 blocks', len(df))
