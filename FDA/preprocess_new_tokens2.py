

import torch
import io
import os
import os.path as osp

from transformers import AutoTokenizer

from tqdm import tqdm
import argparse

from mCLM.tokenizer.molecule_tokenizer import MoleculeTokenizer

from mCLM_tokenizer.tokenizer import convert_SMILES_strings, join_fragments, get_blocks

import pandas as pd

from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol, MurckoScaffoldSmiles


parser = argparse.ArgumentParser(description="Biencoder")

parser.add_argument("--ckpt_path", default="/shared/nas2/shared/llms/mCLM/OnlyBlocks/Top50V2/Qwen2.5-3B_TotalTop1k_splitLoss_1kPretrain/", type=str)
parser.add_argument("--tokenizer_path", default="/shared/nas2/shared/llms/mCLM/OnlyBlocks/Top50V2/Qwen2.5-3B_TotalTop1k_splitLoss_1kPretrain/", type=str)
parser.add_argument("--ckpt", default="latest_checkpoint-epoch=04-step=129000.ckpt", type=str)
parser.add_argument("--GNN_cache", default="", type=str)
parser.add_argument("--out_tokenizer_path", default="./Top1k_FDA2/", type=str)


parser.add_argument("--loss", default="CLIP", type=str)
parser.add_argument("--load_ckpt", default=None, type=str)
parser.add_argument("--load_GNN_ckpt", default=None, type=str)
parser.add_argument("--pretrained_embeddings", default="/home/cne2/data/Chemistry/mCLM_MolCLR/preprocess/Top500/128_dim/", type=str)


parser.add_argument("--seed", default=42, type=int)

parser.add_argument("--base_model", default="/shared/nas2/shared/llms/Qwen2.5-3B/", type=str)
parser.add_argument("--pretrained_text_model", default="/shared/nas2/shared/llms/Qwen2.5-3B/", type=str)
parser.add_argument("--pretrained_tokenizer", default="/shared/nas2/shared/llms/Qwen2.5-3B/", type=str)

parser.add_argument(
    "--freeze_GNN", type=bool, action=argparse.BooleanOptionalAction
)

args = parser.parse_args()


config = vars(args)
print(config)

print('Imports Done')

os.makedirs(config["out_tokenizer_path"], exist_ok=True)


# Define your function here
def is_blocks(mol):
    rv = True

    for mo in mol.split('^'):
        for m in mo.split('.'):
            if m.count('[1*]') > 1 or m.count('[2*]') > 1 or m.count('[3*]') > 1:
                rv = False

            if (not '[1*]' in m) and (not '[2*]' in m) and (not '[3*]' in m):
                rv = False

    return rv

pre_scaff = {}

def get_scaffold(smi):
    if smi in pre_scaff: return pre_scaff[smi]
    try:
        rv = MurckoScaffoldSmiles(smi)
    #except:
    #    return handle_hypervalent(smi)
    except:
        #print('Handling Exception:', smi)
        try:
            if '[PH]' in smi: #handle this weird hypervalent [PH] that occurs a lot 
                rv = MurckoScaffoldSmiles(smi.replace('[PH]', '*')).replace('*', '[PH]')
            elif '[Br-]' in smi:
                return MurckoScaffoldSmiles(smi.replace('[Br-]', '*')).replace('*', '[Br-]')
            elif smi == '[C@@H]12C3(C4C1([Cl-]2)[Cl-]4)C56C37Cl5Cl67': #what a strange molecule
                rv = MurckoScaffoldSmiles(smi.replace('[Cl-]', '*').replace('Cl', '[1*]')).replace('[1*]','Cl').replace('*', '[Cl-]')
            elif '[Cl-]' in smi:
                rv = MurckoScaffoldSmiles(smi.replace('[Cl-]', '*')).replace('*', '[Cl-]')
            else:
                raise Exception #it's something else
        except:
            print('Unhandled Exception:', smi)
            rv = None
    pre_scaff[smi] = rv
    return rv


def load_with_tqdm(file_path, map_location=None, weights_only=True):
    file_size = os.path.getsize(file_path)
    buffer_size = 1024 * 1024  # 1MB chunks
    
    with open(file_path, 'rb') as f, tqdm(desc='Loading', total=file_size, unit='iB', unit_scale=True, unit_divisor=1024) as pbar:
        buffer = bytearray()
        while True:
            chunk = f.read(buffer_size)
            if not chunk:
                break
            buffer.extend(chunk)
            pbar.update(len(chunk))
        
        byte_stream = io.BytesIO(buffer)
        data = torch.load(byte_stream, map_location=map_location, weights_only=weights_only)
    return data

tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
tokenizer.pad_token = tokenizer.eos_token #llama3
tokenizer.add_tokens(['[MOL]', '[/MOL]'])

torch.serialization.add_safe_globals([MoleculeTokenizer])
molecule_tokenizer = load_with_tqdm(config["tokenizer_path"] + "molecule_tokenizer.pth", map_location=torch.device('cpu'), weights_only=False)#torch.load(f)

start_idx = len(tokenizer)
molecule_tokenizer.change_start_idx(start_idx)
molecule_tokenizer.bfloat16 = True

GNN_cache = config["GNN_cache"]

if GNN_cache != "":
    pass

MOL_start = tokenizer.convert_tokens_to_ids('[MOL]')
MOL_end = tokenizer.convert_tokens_to_ids('[/MOL]')

print('Tokenizer Loaded')

df = pd.read_csv('drugs_dedupped.synth.ends.csv')

block_list = df['blocks'].to_list()

blocks = set()

#for mol in tqdm(block_list + scaff_block_list):
for mol in tqdm(block_list ):
    for block in mol.split('^'):
        blocks.add(block)

new_blocks = [nb for nb in blocks if nb not in molecule_tokenizer.block_to_idx]
print('New Blocks:', len(new_blocks))

#print(len(molecule_tokenizer.block_to_idx))
for block in blocks:
    molecule_tokenizer.add_block(block)

print(len(molecule_tokenizer.block_to_idx))
#with open(config['out_tokenizer_path'] + 'molecule_tokenizer.pth', "wb") as f:
#    torch.save(molecule_tokenizer, f)





