import argparse
import os
import os.path as osp
import lightning as L
import copy
import sys
import io
import re
import torch
from lightning import Trainer, seed_everything, Callback
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from mCLM.tokenizer.molecule_tokenizer import MoleculeTokenizer
from mCLM.data.processing import smiles_to_data, smiles_list_to_mols, extract_mol_content

from mCLM_tokenizer.tokenizer import convert_SMILES_strings, join_fragments

from transformers import AutoTokenizer

import pandas as pd
import pickle

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

if __name__ == "__main__":
    torch.cuda.empty_cache()

    #subprocess.run(["nvidia-smi"])

    config = {
        "pretrained_text_model": "michiyasunaga/BioLinkBERT-base",
        "trunc_length": 512,
        "num_warmup_steps": 1000,
        "max_epochs": 2,
        "batch_size": 128,
        "val_batch_size": None,
        "node_dim": 133,
        "edge_dim": 12,
        "hidden_dim_graph": 512,
        "num_mp_layers": 5,
        "num_readout_layers": 1,
        "dropout": 0.13,
        "aggr": "mean",
        "jk": "cat",
        "latent_size": 256,
        "validate_every_n": 1000,
        "lr": 2e-5,
        "data_module": "S1B",
        "ckpt_path": "ckpts/",
        "loss": "CLIP",
        "model": "GNN",
        "load_ckpt": None,
        "seed": 42,
    }

    parser = argparse.ArgumentParser(description="Biencoder")
    parser.add_argument("--trunc_length", default=512, type=int)

    parser.add_argument("--num_warmup_steps", default=1000, type=int)
    parser.add_argument("--max_epochs", default=2, type=int)
    parser.add_argument("--batch_size", default=4, type=int) #2 takes up 29733MiB
    parser.add_argument("--val_batch_size", default=None, type=int)

    parser.add_argument("--node_dim", default=133, type=int)
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
    parser.add_argument("--ckpt_path", default="/shared/nas2/shared/llms/mCLM/Llama-3.2-1B-Instruct-SMolInstruct/", type=str)
    parser.add_argument("--ckpt", default="latest_checkpoint-epoch=00-step=10000.ckpt", type=str)
    parser.add_argument("--loss", default="CLIP", type=str)
    parser.add_argument("--load_ckpt", default=None, type=str)
    parser.add_argument("--load_GNN_ckpt", default=None, type=str)

    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--model", default="mCLM", type=str)

    parser.add_argument("--base_model", default="/shared/nas2/shared/llms/Llama-3.2-1B-Instruct/", type=str)
    parser.add_argument("--pretrained_text_model", default="/shared/nas2/shared/llms/Llama-3.2-1B-Instruct/", type=str)
    parser.add_argument("--pretrained_tokenizer", default="/shared/nas2/shared/llms/Llama-3.2-1B-Instruct/", type=str)    #parser.add_argument(
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

    args = parser.parse_args()

    if args.val_batch_size == None:
        args.val_batch_size = args.batch_size

    config = vars(args)
    print(config)

    if config["val_batch_size"] == None:
        config["val_batch_size"] = config["batch_size"]

    seed_everything(config["seed"])

    print('Imports Done')

    device = 'cpu'#torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #with open(config["ckpt_path"] + "molecule_tokenizer.pth", "rb") as f:
    
    torch.serialization.add_safe_globals([MoleculeTokenizer])
    molecule_tokenizer = load_with_tqdm(config["ckpt_path"] + "molecule_tokenizer.pth", map_location=torch.device('cpu'), weights_only=False)#torch.load(f)

    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    tokenizer.pad_token = tokenizer.eos_token #llama3
    tokenizer.add_tokens(['[MOL]', '[/MOL]'])

    MOL_start = tokenizer.convert_tokens_to_ids('[MOL]')
    MOL_end = tokenizer.convert_tokens_to_ids('[/MOL]')

    print('Tokenizer Loaded')

    model = mCLM(config)

    model.model.extend_text_vocab_size(len(tokenizer.vocab))
    model.model.set_mol_vocab(molecule_tokenizer.GNN_input_map)

    print('Model Created')

    model.load_state_dict(load_with_tqdm(config["ckpt_path"] + config['ckpt'], map_location='cpu')['state_dict'])
    model.to(device)

    print('Model Loaded')

    #GNN_input_map = molecule_tokenizer.GNN_input_map
    #
    #create a dictionary with a version of GNN_input_map for each device (the device is the key)
    #for key in GNN_input_map:
    #    molecule_tokenizer.GNN_input_map[key] = GNN_input_map[key].to(model.device)

    model.train(False)
    model.model.post_training()

    print('Model Set to Inference')

    while True:
        user_input = input("Enter an instruction (type 'quit' to exit): ")
        if user_input == 'quit': break

        mols_list, MOL_input = convert_SMILES_strings(user_input)

        mol_list, cleaned_text = extract_mol_content(MOL_input)

        for mol in mol_list:
            for m in mol.split('^'):
                molecule_tokenizer.add_block(m)
        molecule_tokenizer.create_input()

        messages = [
            {"role": "user", "content": cleaned_text},
        ]
        message_tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        #print(message_tokens)

        frags = [[molecule_tokenizer.get_Idx(m) for m in mol.split('^')] for mol in mol_list]

        message_tokens = torch.Tensor(insert_sublists(message_tokens.squeeze(), frags, MOL_start, MOL_end)).to(torch.int).to(device)
        #print(message_tokens, message_tokens.shape)

        generated = model.generate(
            input_ids=message_tokens.unsqueeze(0),
            max_new_tokens=128,
            num_beams=1,
            do_sample=False,
        )
        #outputs = model.generate(message_tokens, max_new_tokens=128) 
        #out_text = tokenizer.decode(outputs[0])

        #print(generated)

        #extracted_mols = extract_between_MOL(generated[0])
        #print(extracted_mols)
        #extracted_mols = [[molecule_tokenizer.get_block(e) for e in em] for em in extracted_mols]
        #print(extracted_mols)

        message_ids = generated[0, len(message_tokens):]
        #print(message_ids)

        extracted_mols = message_ids > MOL_end
        locs = extracted_mols.nonzero().squeeze()
        #print(locs)

        #print(tokenizer.decode(generated[0].tolist()[len(message_tokens):], skip_special_tokens=True))

        tokens = tokenizer.convert_ids_to_tokens(message_ids)
        #print(tokens)
        tokens = [molecule_tokenizer.get_block(int(message_ids[i]))+'^' if i in locs else t for i, t in enumerate(tokens)]
        #print(tokens)

        message = tokenizer.convert_tokens_to_string(tokens)
        print(message)
        print()
        
        mol_list, message = extract_mol_content(message)
        mol_list = [m[:-1] if m[-1]=='^' else m for m in mol_list]
        mol_list = [Chem.MolToSmiles(join_fragments(smi)) for smi in mol_list]

        #print(mol_list)
        
        message = replace(message, mol_list)

        print(message)
        