import argparse
import os
import os.path as osp
import lightning as L
import copy
import sys
import io
import re
import torch
from torch import nn
import torch.nn.functional as F
from lightning import Trainer, seed_everything, Callback
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from itertools import product
import numpy as np

from rdkit.Chem import Draw
from PIL import Image

from mCLM.tokenizer.molecule_tokenizer import MoleculeTokenizer
from mCLM.data.processing import smiles_to_data, smiles_list_to_mols, extract_mol_content

from mCLM_tokenizer.tokenizer import convert_SMILES_strings, join_fragments

from transformers import AutoTokenizer
from transformers import LogitsProcessorList, LogitsProcessor, AutoTokenizer, AutoModelForCausalLM, NoBadWordsLogitsProcessor
from transformers.utils import logging
logging.set_verbosity_error()

import pandas as pd
import pickle

import matplotlib.pyplot as plt


from tqdm import tqdm

from mCLM.data.dataloaders import KinaseDataModule
from mCLM.model.models import (
    mCLM,
)

from rdkit import Chem

import subprocess

from mCLM.data.processing import insert_sublists, find_first_occurrence


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


def extract_between_MOL(tensor):
    tensor_list = tensor.tolist()  # Convert tensor to list
    extracted = []
    temp = []
    recording = False  # Flag to start recording elements

    for num in tensor_list:
        if num == MOL_start or num == MOL_end:
            if recording and temp:
                extracted.append(temp)  # Save previous group
            temp = []  # Reset temp list
            recording = True  # Start recording after 128256
        elif recording:
            temp.append(num)

    if temp:
        extracted.append(temp)  # Append last collected group if any

    return extracted

def replace(text, mol_list):
    mol_list2 = copy.copy(mol_list)
    def replacer(match):
        if mol_list2:
            return f'<SMILES> {mol_list2.pop(0)} </SMILES>'
        return match.group(0)  # Fallback in case something goes wrong

    restored_text = re.sub(r'\[MOL\](.*?)\[\/MOL\]', replacer, text, count=len(mol_list2))
    return restored_text


class RestrictTokensLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = set(allowed_token_ids)

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float('-inf'))
        allowed_ids = list(self.allowed_token_ids)
        mask[:, allowed_ids] = scores[:, allowed_ids]
        return mask

class ConditionalMolEndProcessor(LogitsProcessor):
    def __init__(self, mol_token_id: int, mol_end_token_id: int, end_token: int, allowed_monoblock_ids: set, allowed_diblock_ids: set):
        self.mol_token_id = mol_token_id
        self.mol_end_token_id = mol_end_token_id
        self.end_token = end_token
        self.allowed_monoblock_ids = allowed_monoblock_ids
        self.allowed_diblock_ids = allowed_diblock_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # input_ids shape: (batch_size, sequence_length)
        batch_size, seq_len = input_ids.shape
        allowed_mono_mask = torch.BoolTensor([j in self.allowed_monoblock_ids for j in range(scores.shape[1])])
        allowed_di_mask = torch.BoolTensor([j in self.allowed_diblock_ids for j in range(scores.shape[1])])

        for i in range(batch_size):
            input_seq = input_ids[i]
            # Find the last occurrence of [MOL] in the sequence
            mol_positions = (input_seq == self.mol_token_id).nonzero(as_tuple=True)[0]
            if len(mol_positions) == 0:
                continue  # [MOL] not found, do nothing
            mol_start = mol_positions[-1].item()  # start counting after last [MOL]
            relative_pos = seq_len - mol_start - 1  # how many tokens after [MOL]

            if relative_pos == 0:
                scores[i, ~allowed_mono_mask] = float("-inf")
            if relative_pos == 1:
                
                scores[i, ~allowed_di_mask] = float("-inf")

            if relative_pos == 2:
                scores[i, ~allowed_mono_mask] = float("-inf")
            if relative_pos == 3:
                scores[i, :] = float("-inf")
                scores[i, self.mol_end_token_id] = 0.0
        return scores



def build_set(blocks):

    mblocks = [b for b in blocks if b in mono_blocks]
    dblocks = [b for b in blocks if b in di_blocks]

    possible_mols = set()

    for start in mblocks:
        for i in range(molecule_size - 2 + 1):
            for dbs in product(dblocks, repeat=i):
                for end in mblocks:
                    new_mol = '^'.join([start] + list(dbs) + [end])

                    possible_mols.add(new_mol)
                    
    return possible_mols





parser = argparse.ArgumentParser(description="Biencoder")
parser.add_argument("--trunc_length", default=512, type=int)

parser.add_argument("--num_warmup_steps", default=1000, type=int)
parser.add_argument("--max_epochs", default=2, type=int)
parser.add_argument("--batch_size", default=4, type=int) #2 takes up 29733MiB
parser.add_argument("--val_batch_size", default=None, type=int)

parser.add_argument("--node_dim", default=138, type=int)
parser.add_argument("--edge_dim", default=12, type=int)
parser.add_argument("--hidden_dim_graph", default=512, type=int)
parser.add_argument("--num_mp_layers", default=5, type=int)
parser.add_argument("--num_readout_layers", default=1, type=int)
parser.add_argument("--dropout", default=0.13, type=float)
parser.add_argument("--aggr", default="mean", type=str)
parser.add_argument("--jk", default="cat", type=str)

parser.add_argument("--latent_size", default=256, type=int)
parser.add_argument("--validate_every_n", default=1000, type=int)
parser.add_argument("--lr", default=5e-5, type=float)

#parser.add_argument("--ckpt_path", default="/shared/nas2/shared/llms/mCLM/OnlyBlocks/Top50V2/Qwen2.5-3B_TotalTop1k_splitLoss_1kPretrain/", type=str)
parser.add_argument("--tokenizer_path", default="/shared/nas2/shared/llms/mCLM/OnlyBlocks/Top50V2/Qwen2.5-3B_TotalTop1k_splitLoss_1kPretrain/", type=str)
#parser.add_argument("--ckpt", default="latest_checkpoint-epoch=04-step=129000.ckpt", type=str)
parser.add_argument("--ckpt_path", default="/shared/nas2/shared/llms/mCLM/OnlyBlocks/Top50V2/Qwen2.5-3B_TotalTop1k_splitLoss_FinetuneV2/predOnlyV2/", type=str)
parser.add_argument("--ckpt", default="best_val_checkpoint.ckpt", type=str)

parser.add_argument("--loss", default="CLIP", type=str)
parser.add_argument("--load_ckpt", default=None, type=str)
parser.add_argument("--load_GNN_ckpt", default=None, type=str)
parser.add_argument("--pretrained_embeddings", default="/home/cne2/data/Chemistry/mCLM_MolCLR/preprocess/Top500/128_dim/", type=str)
parser.add_argument("--GNN_cache", default="", type=str)


parser.add_argument("--seed", default=42, type=int)

parser.add_argument("--model", default="mCLM", type=str)

parser.add_argument("--base_model", default="/shared/nas2/shared/llms/Qwen2.5-3B/", type=str)
parser.add_argument("--pretrained_text_model", default="/shared/nas2/shared/llms/Qwen2.5-3B/", type=str)
parser.add_argument("--pretrained_tokenizer", default="/shared/nas2/shared/llms/Qwen2.5-3B/", type=str)

parser.add_argument(
    "--freeze_GNN", type=bool, action=argparse.BooleanOptionalAction
)

parser.add_argument("--resume_wandb_run", default=None, type=str)

parser.add_argument("--task", default='Kinase', type=str)
parser.add_argument("--weight_decay", default=0.0, type=float)

parser.add_argument("--fold_idx", type=int)

parser.add_argument("--check_val_every_n_steps", default=None, type=int)

parser.add_argument("--no_PEFT", type=bool, action=argparse.BooleanOptionalAction)

args = parser.parse_args()


args.no_PEFT = True

if args.val_batch_size == None:
    args.val_batch_size = args.batch_size

config = vars(args)
print(config)

if config["val_batch_size"] == None:
    config["val_batch_size"] = config["batch_size"]

seed_everything(config["seed"])

print('Imports Done')

device = 'cpu' #torch.device("cuda" if torch.cuda.is_available() else "cpu")

#with open(config["ckpt_path"] + "molecule_tokenizer.pth", "rb") as f:

if not 'molecule_tokenizer' in locals():
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    tokenizer.pad_token = tokenizer.eos_token #llama3
    tokenizer.add_tokens(['[MOL]', '[/MOL]'])

    torch.serialization.add_safe_globals([MoleculeTokenizer])
    molecule_tokenizer = load_with_tqdm(config["tokenizer_path"] + "molecule_tokenizer.pth", map_location=torch.device('cpu'), weights_only=False)#torch.load(f)
    
    start_idx = len(tokenizer)
    molecule_tokenizer.change_start_idx(start_idx)
    molecule_tokenizer.bfloat16 = True

    GNN_cache = config["tokenizer_path"] + 'molecule_tokenizer.graphs.pth'

    if True:
        #Preprocess molecule tokenizer
        if osp.exists(GNN_cache):
            molecule_tokenizer.GNN_input_map = load_with_tqdm(GNN_cache, map_location=torch.device('cpu'), weights_only=False)
        else:
            molecule_tokenizer.create_input()
            with open(GNN_cache, "wb") as f:
                torch.save(molecule_tokenizer.GNN_input_map, f)


    MOL_start = tokenizer.convert_tokens_to_ids('[MOL]')
    MOL_end = tokenizer.convert_tokens_to_ids('[/MOL]')

    library = set()
    for block in molecule_tokenizer.block_to_idx:
        library.add(block)

    di_blocks = set([b for b in library if '1*' in b and '2*' in b])
    mono_blocks = set([b for b in library if ('3*' in b)])

    total_blocks = list(mono_blocks.union(di_blocks))

    print('Tokenizer Loaded')

if not 'model' in locals():

    model = mCLM(config)


    if config['pretrained_embeddings'] != None:
        pretrain_mol_embeddings = torch.load(config['pretrained_embeddings'] + 'precomputed_tokens.pt').to(torch.bfloat16)
        pretrain_mol_embeddings = nn.Embedding.from_pretrained(pretrain_mol_embeddings, freeze=True)
        pretrain_mol_embeddings.weight.data = pretrain_mol_embeddings.weight.data.to(torch.bfloat16)

        
        model.model.finalize_molecule_embeddings(embeddings=pretrain_mol_embeddings)
        model.model.use_mol_embeddings(True)

    model.model.extend_text_vocab_size(len(tokenizer.vocab))
    model.model.set_mol_vocab(molecule_tokenizer.GNN_input_map)

    model = model.to(torch.bfloat16)

    print('Model Created')

    if config['ckpt'] != None:
        sd = load_with_tqdm(config["ckpt_path"] + config['ckpt'], map_location='cpu')['state_dict']
        #ignore GNN keys
        for key in list(sd.keys()):
            if 'mol_gnn' in key: sd.pop(key)

        model.load_state_dict(sd, strict=False)
    model.to(device)

    print('Model Loaded')

    #GNN_input_map = molecule_tokenizer.GNN_input_map
    #
    #create a dictionary with a version of GNN_input_map for each device (the device is the key)
    #for key in GNN_input_map:
    #    molecule_tokenizer.GNN_input_map[key] = GNN_input_map[key].to(model.device)

    model.train(False)
    #model.model.post_training(1024)

    print('Model Set to Inference')

bad_words_ids = None


def message_ids_to_string(message_ids, add_mol=False):
    
    extracted_mols = message_ids > MOL_end
    locs = extracted_mols.nonzero().squeeze()
    #print(locs)

    #print(tokenizer.decode(generated[0].tolist()[len(message_tokens):], skip_special_tokens=True))

    tokens = tokenizer.convert_ids_to_tokens(message_ids, skip_special_tokens=True)
    #print(tokens)
    tokens = [molecule_tokenizer.get_block(int(message_ids[i]))+'^' if i in locs else t for i, t in enumerate(tokens)]
    #print(tokens)

    #print(tokens)
    #print()

    message = '[MOL]'*add_mol + tokenizer.convert_tokens_to_string(tokens)
    #print(message)
    #print()
    
    mol_list, smi_message = extract_mol_content(message)
    mol_list = [m[:-1] if m[-1]=='^' else m for m in mol_list]
    joined_mol_list = [Chem.MolToSmiles(join_fragments(smi)) for smi in mol_list]

    #print(mol_list)
    
    smi_message = replace(smi_message, mol_list)

    return message, smi_message, mol_list


def get_molecule(user_input, allowed_token_ids, allowed_blocks=None, num_mols=5, molecule_size=3, bad_word_ids=None):
    #user_input = input("Enter an instruction (type 'quit' to exit): ")
    #if user_input == 'quit': break

    if allowed_token_ids == None:
        assert allowed_blocks != None
        allowed_token_ids = [molecule_tokenizer.get_Idx(b) for b in allowed_blocks]

    mols_list, MOL_input = convert_SMILES_strings(user_input)

    mol_list, cleaned_text = extract_mol_content(MOL_input)

    new_blocks = []

    for mol in mol_list:
        for m in mol.split('^'):
            new_blocks.append(m)
            #molecule_tokenizer.add_block(m)
    new_blocks = [nb for nb in new_blocks if nb not in molecule_tokenizer.block_to_idx]
    #print('New Blocks:', len(new_blocks))
    #if len(new_blocks) > 0: break
    #molecule_tokenizer.create_input_from_list(new_blocks)

    messages = [
        {"role": "user", "content": cleaned_text},
    ]
    message_tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    #print(message_tokens)

    frags = [[molecule_tokenizer.get_Idx(m) for m in mol.split('^')] for mol in mol_list]
    
    message_tokens = torch.Tensor(insert_sublists(message_tokens.squeeze(), frags, MOL_start, MOL_end)+[MOL_start]).to(torch.int).to(device)
    #print(message_tokens, message_tokens.shape)
    
    processors = [ RestrictTokensLogitsProcessor(allowed_token_ids + [MOL_end]), \
        ConditionalMolEndProcessor(MOL_start, MOL_end, tokenizer.convert_tokens_to_ids(tokenizer.eos_token), \
            [molecule_tokenizer.get_Idx(b) for b in mono_blocks], \
            [molecule_tokenizer.get_Idx(b) for b in di_blocks]), \
        ]
        
    if len(bad_word_ids) > 0: 
        processors.append(NoBadWordsLogitsProcessor(bad_word_ids))
        
    processor = LogitsProcessorList(processors)
        

    generated = model.generate(
        input_ids=message_tokens.long().unsqueeze(0),
        max_new_tokens=molecule_size+1,
        num_beams=num_mols,
        num_return_sequences=num_mols,
        do_sample=False,
        bad_words_ids=bad_words_ids,
        diversity_penalty=1.0,
        num_beam_groups=num_mols,
        logits_processor=processor,
    )
    
    all_mol_list = []

    for i in range(num_mols):
        message_ids = generated[i, len(message_tokens):]
        
        mol_msg, smiles_msg, mol_list = message_ids_to_string(message_ids, add_mol=True)

        all_mol_list.append(mol_list)
        
    mols = [mol_list[0] for mol_list in all_mol_list]

    return [[m for m in mol.split('^')] for mol in mols], [[molecule_tokenizer.get_Idx(m) for m in mol.split('^')] for mol in mols], mols

        
def get_score(user_input):
    #user_input = input("Enter an instruction (type 'quit' to exit): ")
    #if user_input == 'quit': break

    allowed_token_ids = [tokenizer('No')['input_ids'][0], tokenizer('Yes')['input_ids'][0]]

    mols_list, MOL_input = convert_SMILES_strings(user_input)

    mol_list, cleaned_text = extract_mol_content(MOL_input)

    new_blocks = []

    for mol in mol_list:
        for m in mol.split('^'):
            new_blocks.append(m)
            #molecule_tokenizer.add_block(m)
    new_blocks = [nb for nb in new_blocks if nb not in molecule_tokenizer.block_to_idx]
    #print('New Blocks:', len(new_blocks))
    #if len(new_blocks) > 0: break
    #molecule_tokenizer.create_input_from_list(new_blocks)

    messages = [
        {"role": "user", "content": cleaned_text},
    ]
    message_tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    #print(message_tokens)

    frags = [[molecule_tokenizer.get_Idx(m) for m in mol.split('^')] for mol in mol_list]
    
    message_tokens = torch.Tensor(insert_sublists(message_tokens.squeeze(), frags, MOL_start, MOL_end)).to(torch.int).to(device)
    #print(message_tokens, message_tokens.shape)
    
    processor = LogitsProcessorList([RestrictTokensLogitsProcessor(allowed_token_ids)])


    output = model.generate(
        input_ids=message_tokens.unsqueeze(0),
        max_new_tokens=1,
        num_beams=1,
        do_sample=False,
        bad_words_ids=bad_words_ids,
        logits_processor=processor,
        return_dict_in_generate=True,
        output_scores=True,
    )

    # Get generated token IDs (excluding prompt)
    #generated_tokens = output.sequences[0][message_tokens.shape[-1]:]

    logits = output.scores[0][0][allowed_token_ids]

    #print(logits)

    return F.softmax(logits, dim=0)[1] #Yes


