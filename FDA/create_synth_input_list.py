
import pandas as pd
import ast

seed = 42

df = pd.read_csv('drugs_processed.csv')
#df['blocks'] = df['blocks'].apply(ast.literal_eval)

#df = df[df['blocks'].apply(lambda x : len(x)>0)]

#df = df[df['blocks'].apply(lambda x : len(x[0].split('^'))>2)]

print(len(df))

df = df.drop_duplicates(subset=['SMILES'], keep='first')

print(len(df))


all_synth_blocks = set([s.strip().lower() for s in open('/shared/nas/data/m1/shared-resource/MoleculeLanguage/mCLM/synth_blocks_top5000000k.txt').readlines()])
#print(len(all_synth_blocks))
#print(list(all_synth_blocks)[:10])

#zz

def get_synth(mols):
    for m in mols:
        for block in m.split('^'):
            if not block.lower() in all_synth_blocks:
                print(block)
                return False
            else:
                print('here', block)
    return True

#print(df['blocks'].iloc[0])


#df = df[df['blocks'].apply(get_synth)]

#print(len(df))



#names = df['Name'].tolist()
smiles = df['SMILES'].tolist()
#blocks = df['blocks'].tolist()



def write_strings_to_file(strings: set, file_path: str):
    with open(file_path, "w") as f:
        for s in strings:
            f.write(s + "\n")

write_strings_to_file(smiles, 'drugs_dedupped.csv')

