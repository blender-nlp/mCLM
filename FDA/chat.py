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

import ast

import pickle

from collections import defaultdict

from scipy.stats import wasserstein_distance

from rdkit.Chem import Draw
from PIL import Image, ImageDraw, ImageFont

from mCLM.tokenizer.molecule_tokenizer import MoleculeTokenizer
from mCLM.data.processing import smiles_to_data, smiles_list_to_mols, extract_mol_content

from mCLM_tokenizer.tokenizer import convert_SMILES_strings, join_fragments

from transformers import AutoTokenizer
from transformers import LogitsProcessorList, LogitsProcessor, AutoTokenizer, AutoModelForCausalLM
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

# Add the src directory to sys.path
sys.path.append(os.path.abspath('../admet_oracle_model/src'))

from admet_oracle_model.src.main import prepare_dataset, evaluate, MLP

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
        allowed_mask = torch.BoolTensor([j in self.allowed_monoblock_ids or j == self.mol_end_token_id for j in range(scores.shape[1])])

        for i in range(batch_size):
            input_seq = input_ids[i]
            # Find the last occurrence of [MOL] in the sequence
            mol_positions = (input_seq == self.mol_token_id).nonzero(as_tuple=True)[0]
            if len(mol_positions) == 0:
                continue  # [MOL] not found, do nothing
            mol_start = mol_positions[-1].item()  # start counting after last [MOL]
            relative_pos = seq_len - mol_start - 1  # how many tokens after [MOL]

            if relative_pos == 0:
                scores[i, ~allowed_mask] = float("-inf")

            if input_ids[i, -1] in self.allowed_diblock_ids:
                scores[i, self.mol_end_token_id] = float("-inf")

            if input_ids[i, -1] in self.allowed_diblock_ids and relative_pos == 5:
                scores[i, ~allowed_mask] = float("-inf")

            if input_ids[i, -1] in self.allowed_monoblock_ids and relative_pos != 1:
                scores[i, :] = float("-inf")
                scores[i, self.mol_end_token_id] = 0.0
                

            if input_ids[i, -1] in self.allowed_monoblock_ids and relative_pos == 1:
                scores[i, self.mol_end_token_id] = float("-inf")

            if input_ids[i, -1] == self.mol_end_token_id:
                mask = torch.ones_like(scores[i,:], dtype=bool)
                mask[tokenizer.pad_token_id] = False
                #mask[self.mol_token_id] = False
                scores[i, ~mask] = float("-inf")
                scores[i, tokenizer.pad_token_id] = 0.0


        return scores




def oracle_score(data_path, task):
    ckpt_path = '../admet_oracle_model/checkpoints'

    task_ckpt_path = os.path.join(ckpt_path, f'{task}_mlp.pt')

    dataloader, smiles = prepare_dataset(data_path, ckpt_path)
    model = MLP().to(device)

    model.load_state_dict(torch.load(task_ckpt_path))
    all_preds = evaluate(model, dataloader, device).squeeze()

    return all_preds



def smiles_to_image(smiles_list, img_size=(200, 200), grid_size=None):
    """
    Generates an image for each SMILES string and combines them into one.
    :param smiles_list: List of SMILES strings
    :param img_size: Tuple (width, height) for each molecule image
    :param grid_size: Tuple (rows, cols) for arranging images; if None, it will be determined automatically
    :return: Combined image
    """
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    valid_mols = [mol for mol in mols if mol is not None]
    
    if not valid_mols:
        raise ValueError("No valid molecules generated from SMILES.")
    
    # Determine grid size automatically if not provided
    n = len(valid_mols)
    if grid_size is None:
        cols = int(n**0.5) + 1
        rows = (n + cols - 1) // cols  # Ensure enough rows
    else:
        rows, cols = grid_size
    
    # Draw individual molecule images
    images = [Draw.MolToImage(mol, size=img_size) for mol in valid_mols]
    
    # Create a blank canvas
    total_width = cols * img_size[0]
    total_height = rows * img_size[1]
    combined_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    
    # Paste images onto the canvas
    for idx, img in enumerate(images):
        row, col = divmod(idx, cols)
        x_offset = col * img_size[0]
        y_offset = row * img_size[1]
        combined_image.paste(img, (x_offset, y_offset))
    
    return combined_image


def smiles_to_labeled_row(smiles, name, modifications, score_change, img_size=(200, 200), font_path=None, font_size=14, save_path=None):
    """
    Generates a horizontal row of molecule images with labels and score changes underneath.

    :param smiles: Original SMILES string
    :param name: Name of the original molecule
    :param modifications: Dict of property -> SMILES string
    :param score_change: Dict of property -> (before, after) floats
    :param img_size: Size of each molecule image (width, height)
    :param font_path: Optional path to a TTF font
    :param font_size: Font size
    :param save_path: Optional file path to save the image
    :return: Combined PIL image
    """
    try:
        font = ImageFont.truetype(font_path if font_path else "arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # Prepare list of (mol, label lines)
    entries = [(
        Chem.MolFromSmiles(smiles),
        ["original", name]
    )]

    for prop, mod_smi in modifications.items():
        mol = Chem.MolFromSmiles(mod_smi)
        if mol is None:
            continue
        before, after = score_change.get(prop, (None, None))
        label_lines = [prop]
        if before is not None and after is not None:
            label_lines.append(f"Before: {before:.3f}")
            label_lines.append(f"After:  {after:.3f}")
        entries.append((mol, label_lines))

    if not entries:
        raise ValueError("No valid molecules to draw.")

    # Render molecule images
    mol_images = [Draw.MolToImage(mol, size=img_size) for mol, _ in entries]

    # Determine max label height
    label_heights = []
    for _, label_lines in entries:
        line_spacing = 4  # Adjust as needed for spacing between lines
        line_heights = [font.getbbox(line)[3] - font.getbbox(line)[1] for line in label_lines]
        height = sum(line_heights) + line_spacing * (len(label_lines) - 1)
        label_heights.append(height)
    max_label_height = max(label_heights) + 10  # Add spacing
    

    # Create final image
    total_width = len(entries) * img_size[0]
    total_height = img_size[1] + max_label_height
    combined = Image.new("RGB", (total_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(combined)

    # Paste images and draw text
    for i, (img, (_, label_lines)) in enumerate(zip(mol_images, entries)):
        x_offset = i * img_size[0]
        combined.paste(img, (x_offset, 0))
        y_text = img_size[1] + 2
        for line in label_lines:
            bbox = font.getbbox(line)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            draw.text((x_offset + (img_size[0] - w) // 2, y_text), line, fill="black", font=font)
            y_text += h + line_spacing


    if save_path:
        combined.save(save_path)

    return combined

if __name__ == "__main__":
    torch.cuda.empty_cache()


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
    #parser.add_argument("--ckpt_path", default="/shared/nas2/shared/llms/mCLM/Total_mCLM_Qwen2.5-0.5B_NoGNN_FastV2_Shrink25k_OnlyBlocks2_splitLR/", type=str)
    parser.add_argument("--tokenizer_path", default="/shared/nas2/shared/llms/mCLM/OnlyBlocks/Top50V2/Qwen2.5-3B_TotalTop1k_splitLoss_1kPretrain/", type=str)
    #parser.add_argument("--ckpt", default="latest_checkpoint-epoch=00-step=30000.ckpt", type=str)

    #parser.add_argument("--ckpt_path", default="/shared/nas2/shared/llms/mCLM/OnlyBlocks/Top50V2/Qwen2.5-3B_TotalTop1k_splitLoss_1kPretrain/", type=str)
    #parser.add_argument("--tokenizer_path", default="./Top1k_FDA/", type=str)
    #parser.add_argument("--ckpt", default="latest_checkpoint-epoch=04-step=129000.ckpt", type=str)
    parser.add_argument("--ckpt_path", default="/shared/nas2/shared/llms/mCLM/OnlyBlocks/Top50V2/Qwen2.5-3B_TotalTop1k_splitLoss_FinetuneV2/posneg/", type=str)
    #parser.add_argument("--tokenizer_path", default="./Top1k_FDA/", type=str)
    parser.add_argument("--ckpt", default="best_val_checkpoint.ckpt", type=str)
    
    parser.add_argument("--loss", default="CLIP", type=str)
    parser.add_argument("--load_ckpt", default=None, type=str)
    parser.add_argument("--load_GNN_ckpt", default=None, type=str)
    parser.add_argument("--pretrained_embeddings", default="/home/cne2/data/Chemistry/mCLM_MolCLR/preprocess/Top500/FDA/128_dim/", type=str)
    parser.add_argument("--GNN_cache", default="", type=str)


    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--model", default="mCLM", type=str)

    #parser.add_argument("--base_model", default="/shared/nas2/shared/llms/Llama-3.2-1B-Instruct/", type=str)
    #parser.add_argument("--pretrained_text_model", default="/shared/nas2/shared/llms/Llama-3.2-1B-Instruct/", type=str)
    #parser.add_argument("--pretrained_tokenizer", default="/shared/nas2/shared/llms/Llama-3.2-1B-Instruct/", type=str)    #parser.add_argument(
    #parser.add_argument("--base_model", default="/shared/nas2/shared/llms/Qwen2.5-0.5B/", type=str)
    #parser.add_argument("--pretrained_text_model", default="/shared/nas2/shared/llms/Qwen2.5-0.5B/", type=str)
    #parser.add_argument("--pretrained_tokenizer", default="/shared/nas2/shared/llms/Qwen2.5-0.5B/", type=str)
    #parser.add_argument("--base_model", default="/shared/nas2/shared/llms/Qwen3-0.6B-Base/", type=str)
    #parser.add_argument("--pretrained_text_model", default="/shared/nas2/shared/llms/Qwen3-0.6B-Base/", type=str)
    #parser.add_argument("--pretrained_tokenizer", default="/shared/nas2/shared/llms/Qwen3-0.6B-Base/", type=str)
    parser.add_argument("--base_model", default="/shared/nas2/shared/llms/Qwen2.5-3B/", type=str)
    parser.add_argument("--pretrained_text_model", default="/shared/nas2/shared/llms/Qwen2.5-3B/", type=str)
    parser.add_argument("--pretrained_tokenizer", default="/shared/nas2/shared/llms/Qwen2.5-3B/", type=str)

    #    "--freeze_text_encoder", type=bool, action=argparse.BooleanOptionalAction
    #)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        for key in molecule_tokenizer.GNN_input_map:
            molecule_tokenizer.GNN_input_map[key] = molecule_tokenizer.GNN_input_map[key].to(device)

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

    if False:
        bad_words_ids = [[int(s.strip())] for s in open('bad_tokens.txt').readlines()]
        #bad_words_ids = [[tokenizer(s)['input_ids'][0]] for s in bad_words_ids]


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


    def get_molecule(instruct, response, allowed_token_ids):
        #user_input = input("Enter an instruction (type 'quit' to exit): ")
        #if user_input == 'quit': break

        #mols_list, MOL_input = convert_SMILES_strings(user_input)

        #print(instruct, response)

        mol_list1, cleaned_instruct = extract_mol_content(instruct)
        mol_list2, cleaned_response = extract_mol_content(response)

        mol_list = mol_list1 + mol_list2


        messages = [
            {"role": "user", "content": cleaned_instruct},
            {"role": "assistant", "content": cleaned_response + '[MOL]'},
        ]
        message_tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")

        flat_x = message_tokens.flatten()
        indices = (flat_x == MOL_start).nonzero(as_tuple=True)[0]
        last_index = indices[-1].item()
        message_tokens = message_tokens[0,:last_index]
        #print(message_tokens, message_tokens.shape)
        #print([tokenizer.decode(t) for t in message_tokens])
        #zz

        frags = [[molecule_tokenizer.get_Idx(m) for m in mol.split('^')] for mol in mol_list]
        
        message_tokens = torch.Tensor(insert_sublists(message_tokens.squeeze(), frags, MOL_start, MOL_end)+[MOL_start]).to(torch.int).to(device)
        #print(message_tokens, message_tokens.shape)
        
        processor = LogitsProcessorList([ RestrictTokensLogitsProcessor(allowed_token_ids + [MOL_end, tokenizer.pad_token_id]), \
            ConditionalMolEndProcessor(MOL_start, MOL_end, tokenizer.convert_tokens_to_ids(tokenizer.eos_token), \
                [molecule_tokenizer.get_Idx(b) for b in mono_blocks], \
                [molecule_tokenizer.get_Idx(b) for b in di_blocks]), \
            ])
            
        #print(model.device, message_tokens.long().unsqueeze(0).device)
        generated = model.generate(
            input_ids=message_tokens.long().unsqueeze(0),
            max_new_tokens=10,
            num_beams=5,
            do_sample=False,
            logits_processor=processor,
        )
        
        all_mol_list = []

        message_ids = generated[0, len(message_tokens):] 
        #print(message_ids)
        tokens = tokenizer.convert_ids_to_tokens(message_ids, skip_special_tokens=True)
        tokens = [molecule_tokenizer.get_block(int(message_ids[i]))+'^' if t == None else t for i, t in enumerate(tokens)]
        print(tokens)
        #zz
        mol_msg, smiles_msg, mol_list = message_ids_to_string(message_ids, add_mol=True)

        #print('ml:', mol_list)
        #print(smiles_msg)
        #print()
        all_mol_list.append(mol_list)
        #print(all_mol_list)
        mols = [mol_list[0] for mol_list in all_mol_list]

        return [[molecule_tokenizer.get_Idx(m) for m in mol.split('^')] for mol in mols], mols

    

    templates = {
        'ames':'Generate a molecule that has lower mutagenicity on the ames test than [MOL] [/MOL].',
        'bbbp':'Generate a molecule that has higher blood brain barrier permeability than [MOL] [/MOL].',
        'cyp3a4':'Generate a molecule that has lower CYP3A4 inhibitory activity than [MOL] [/MOL].',
        'dili':'Generate a molecule that causes less drug-induced liver injury than [MOL] [/MOL].',
        'hia':'Generate a molecule that has higher Human Intestinal Absorption than [MOL] [/MOL].',
        'pgp':"Generate a molecule that has lower PGP inhibition than [MOL] [/MOL].",
    }
    

    tasks = ['pgp']#['ames', 'bbbp', 'cyp3a4', 'dili', 'hia', 'pgp']


    seed = 42

    df = pd.read_csv('drugs_processed.csv')
    df['blocks'] = df['blocks'].apply(ast.literal_eval)

    df = df[df['blocks'].apply(lambda x : len(x)>0)]
    
    df = df[df['blocks'].apply(lambda x : len(x[0].split('^'))>2)]
    
    #df = df.head(50)


    print(df)

    names = df['Name'].tolist()
    smiles = df['SMILES'].tolist()
    blocks = df['blocks'].tolist()


    if True:
        
        print(len(library))

        OOD_mols = 0

        all_done = set()
        new_blocks = []
        for smi, blcks in zip(smiles, blocks):
            if smi in all_done: continue
            all_done.add(smi)
            new_blocks.append(blcks)

        new_library = set()
        for blcks in new_blocks:
            flag = False
            for block in blcks:
                for b in block.split('^'):
                    if b not in library:
                        new_library.add(b)
                        flag = True
            OOD_mols += flag
        print(len(new_library))

        print(OOD_mols)

        zz

    new_smiles = defaultdict(list)
    all_blocks = defaultdict(list)

    scores = defaultdict(list)

    for task in tqdm(tasks):

        checkpoint_file = f'progress_checkpoint_{task}.pkl'
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
                new_smiles = checkpoint['new_smiles']
                all_blocks = checkpoint['all_blocks']
                completed_smis = checkpoint['completed_smis']
            print("Checkpoint loaded.")
        else:
            new_smiles = {task: []}
            all_blocks = {task: []}
            completed_smis = set()
            print("Starting fresh.")



        for smi, blck in tqdm(zip(smiles, blocks), total=len(blocks)):
            if smi in completed_smis:
                continue

            inst = templates[task].replace('[MOL] [/MOL]', ''.join(['[MOL]' + b + '[/MOL]' for b in blck]))
            resp = '[MOL]'

            token_ids, mols = get_molecule(inst, resp, allowed_token_ids=list(molecule_tokenizer.idx_to_block.keys()))
            
            inp = mols

            if len(inp) != 0:
                new_smis = '.'.join([Chem.MolToSmiles(join_fragments(mol)) for mol in inp])

                new_smiles[task].append(new_smis)
                all_blocks[task].append((blck, '.'.join(inp)))
            else:
                print(task,smi, blck, 'failed')
                zz

            # Mark as completed and save progress
            completed_smis.add(smi)
            with open(checkpoint_file, 'wb') as f:
                pickle.dump({
                    'new_smiles': new_smiles,
                    'all_blocks': all_blocks,
                    'completed_smis': completed_smis
                }, f)



        comb_scores = oracle_score(smiles + new_smiles[task], task)
        pre_scores = comb_scores[:len(smiles)]
        after_scores = comb_scores[len(smiles):]

        print(task + ' before:', np.mean(pre_scores))
        print(task + ' after:', np.mean(after_scores))

        scores[task].append((pre_scores, after_scores))
        
    #print(scores)
    #print(smiles)
    #print(names)
    #print(new_smiles)

    data_to_save = {
        'scores': scores,
        'smiles': smiles,
        'names': names,
        'new_smiles': new_smiles,
        'all_blocks': all_blocks,
    }

    with open(f'saved_improve_{tasks[0]}.pkl', 'wb') as f:
        pickle.dump(data_to_save, f)

    for i, (name, smi) in enumerate(zip(names, smiles)):
        name = name.replace('<n>', '').replace('</n>', '')
        modifications= {task:new_smiles[task][i] for task in tasks}
        score_change= {task:(scores[task][0][0][i], scores[task][0][1][i]) for task in tasks}
        img = smiles_to_labeled_row(smi, name, modifications, score_change, save_path=f'images/{tasks[0]}/{name}.png')
