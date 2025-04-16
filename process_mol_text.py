

import os.path as osp
import os
import csv 

import pandas as pd

from mCLM.data.processing import extract_mol_content

from tqdm import tqdm
tqdm.pandas()


def extract_mol_content2(instruction, response):
    try:
        ml1, clean_instruction = extract_mol_content(instruction)
    except:
        print(instruction)
        zz
    try:
        ml2, clean_response = extract_mol_content(response)
    except:
        print(response)
        zz
    return ml1 + ml2, clean_instruction.replace('\\n', '\n').strip(), clean_response.replace('\\n', '\n').strip()



instruction_data_path = '/shared/nas/data/m1/shared-resource/MoleculeLanguage/mCLM/instruction/processed/'
synthetic_data_path = '/shared/nas/data/m1/shared-resource/MoleculeLanguage/mCLM/synthetic/processed/'

if False:
    ddir = osp.join(instruction_data_path)
    f = 'SMolInstruct_train.csv'
    df = pd.read_csv(osp.join(ddir, f), dtype={'instruction': str, 'response': str})
    print(f)
    print(df)
    df[['mol_list', 'cleaned_instruction', 'cleaned_response']] = df.progress_apply(lambda x: pd.Series(extract_mol_content2(x['instruction'], x['response'])), axis=1)
    df.to_csv(osp.join(ddir.replace('processed', 'dataloader_processed'), f), index=False)

    ddir = osp.join(instruction_data_path)
    f = 'SMolInstruct_val.csv'
    df = pd.read_csv(osp.join(ddir, f), dtype={'instruction': str, 'response': str})
    print(f)
    print(df)
    df[['mol_list', 'cleaned_instruction', 'cleaned_response']] = df.progress_apply(lambda x: pd.Series(extract_mol_content2(x['instruction'], x['response'])), axis=1)
    df.to_csv(osp.join(ddir.replace('processed', 'dataloader_processed'), f), index=False)

    ddir = osp.join(instruction_data_path)
    f = 'SMolInstruct_test.csv'
    df = pd.read_csv(osp.join(ddir, f), dtype={'instruction': str, 'response': str})
    print(f)
    print(df)
    df[['mol_list', 'cleaned_instruction', 'cleaned_response']] = df.progress_apply(lambda x: pd.Series(extract_mol_content2(x['instruction'], x['response'])), axis=1)
    df.to_csv(osp.join(ddir.replace('processed', 'dataloader_processed'), f), index=False)


for subdir in ['synthetic_chembl', 'synthetic_admet_chembl','mCLM', 'pos_neg', 'pos_neg', 'pos_neg', 'pos_pos', 'property_to_mol','multi_property_to_mol', 'mol_only','regression', 'classification']:
    ddir = osp.join(synthetic_data_path, subdir)
    os.makedirs(ddir.replace('processed', 'dataloader_processed'), exist_ok=True)
    files = [f for f in os.listdir(ddir) if os.path.isfile(os.path.join(ddir, f))]
    for f in tqdm(files):
        print(f)
        newf = osp.join(ddir.replace('processed', 'dataloader_processed'), f)
        if osp.exists(newf): continue
        df = pd.read_csv(osp.join(ddir, f), dtype={'instruction': str, 'response': str})
        #print(df)
        if len(df) == 0: continue
        df[['mol_list', 'cleaned_instruction', 'cleaned_response']] = df.progress_apply(lambda x: pd.Series(extract_mol_content2(x['instruction'], x['response'])), axis=1)
        df.to_csv(newf, index=False)



