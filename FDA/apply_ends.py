import os
import os.path as osp
import pandas as pd
import ast
from shutil import copytree


def replace_ends(mol_list):

    new_mol_list = []
    for mol in mol_list:
        new_react = []
        for mo in mol.split('.'):
            
            split = mo.split('^')
            new_mol = []
            for m in split:
                if ('[1*]' in m) and (not '[2*]' in m): new_mol.append(m.replace('[1*]', '[3*]'))
                elif (not '[1*]' in m) and ('[2*]' in m): new_mol.append(m.replace('[2*]', '[3*]'))
                else: new_mol.append(m)

            new_react.append('^'.join(new_mol))
        new_mol_list.append('.'.join(new_react))

    return new_mol_list


def process_csv_files(input_path, output_path):

    #try:
    df = pd.read_csv(
        input_path,
        dtype={
            'SMILES': str,
            'blocks': str,
        },
        keep_default_na=False,
        na_values=[]
    )
    if 'blocks' in df:
        #df['blocks'] = df['blocks'].apply(ast.literal_eval)
        df['blocks'] = df['blocks'].apply(lambda x : replace_ends([x])[0])
    df.to_csv(output_path, index=False)
    print(f"Processed and saved: {output_path}")
    #except Exception as e:
    #    print(f"Failed to process {input_path}: {e}")

# Example usage
process_csv_files("drugs_dedupped.synth.csv", "drugs_dedupped.synth.ends.csv")




