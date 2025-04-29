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
from lightning import Trainer, seed_everything, Callback
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger


from mCLM.tokenizer.molecule_tokenizer import MoleculeTokenizer
from mCLM.data.processing import smiles_to_data, smiles_list_to_mols, extract_mol_content

from mCLM_tokenizer.tokenizer import convert_SMILES_strings, join_fragments

from transformers import AutoTokenizer
from transformers import LogitsProcessorList, LogitsProcessor, AutoTokenizer, AutoModelForCausalLM

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
    #parser.add_argument("--tokenizer_path", default="/shared/nas2/shared/llms/mCLM/Qwen2.5-0.5B_SMolInstruct_NoGNN_OnlyBlocks2_lowLR/", type=str)
    #parser.add_argument("--ckpt", default="latest_checkpoint-epoch=00-step=30000.ckpt", type=str)

    parser.add_argument("--ckpt_path", default="/shared/nas2/shared/llms/mCLM/Qwen2.5-0.5B_SMolInstructTop50k_NoGNN_splitLR_splitLoss/", type=str)
    parser.add_argument("--tokenizer_path", default="/shared/nas2/shared/llms/mCLM/Qwen2.5-0.5B_SMolInstructTop50k_NoGNN_splitLR_splitLoss/", type=str)
    parser.add_argument("--ckpt", default="latest_checkpoint-epoch=01-step=170000.ckpt", type=str)
    
    parser.add_argument("--loss", default="CLIP", type=str)
    parser.add_argument("--load_ckpt", default=None, type=str)
    parser.add_argument("--load_GNN_ckpt", default=None, type=str)
    parser.add_argument("--pretrained_embeddings", default="/home/cne2/data/Chemistry/mCLM_MolCLR/preprocess/Top50k/128_dim/", type=str)
    parser.add_argument("--GNN_cache", default="", type=str)


    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--model", default="mCLM", type=str)

    #parser.add_argument("--base_model", default="/shared/nas2/shared/llms/Llama-3.2-1B-Instruct/", type=str)
    #parser.add_argument("--pretrained_text_model", default="/shared/nas2/shared/llms/Llama-3.2-1B-Instruct/", type=str)
    #parser.add_argument("--pretrained_tokenizer", default="/shared/nas2/shared/llms/Llama-3.2-1B-Instruct/", type=str)    #parser.add_argument(
    parser.add_argument("--base_model", default="/shared/nas2/shared/llms/Qwen2.5-0.5B/", type=str)
    parser.add_argument("--pretrained_text_model", default="/shared/nas2/shared/llms/Qwen2.5-0.5B/", type=str)
    parser.add_argument("--pretrained_tokenizer", default="/shared/nas2/shared/llms/Qwen2.5-0.5B/", type=str)
    #parser.add_argument("--base_model", default="/shared/nas2/shared/llms/Qwen2.5-7B/", type=str)
    #parser.add_argument("--pretrained_text_model", default="/shared/nas2/shared/llms/Qwen2.5-7B/", type=str)
    #parser.add_argument("--pretrained_tokenizer", default="/shared/nas2/shared/llms/Qwen2.5-7B/", type=str)

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

    if False:
        bad_words_ids = []
        for block in molecule_tokenizer.block_to_idx:
            if block.count('[1*]') > 1 or block.count('[2*]') > 1 or block.count('[3*]') > 1:
                bad_words_ids.append([molecule_tokenizer.block_to_idx[block]])

        if False:
            class SuppressTokensProcessor(LogitsProcessor):
                def __init__(self, banned_token_ids):
                    self.banned_token_ids = set(banned_token_ids)

                def __call__(self, input_ids, scores):
                    scores[:, list(self.banned_token_ids)] = -float("inf")
                    return scores

            logits_processor = LogitsProcessorList([
                SuppressTokensProcessor(bad_words_ids)
            ])
    

    while True:
        user_input = input("Enter an instruction (type 'quit' to exit): ")
        if user_input == 'quit': break

        if True: #try:

            mols_list, MOL_input = convert_SMILES_strings(user_input)

            mol_list, cleaned_text = extract_mol_content(MOL_input)

            new_blocks = []

            for mol in mol_list:
                for m in mol.split('^'):
                    new_blocks.append(m)
                    #molecule_tokenizer.add_block(m)
            new_blocks = [nb for nb in new_blocks if nb not in molecule_tokenizer.block_to_idx]
            print('New Blocks:', len(new_blocks))
            if len(new_blocks) > 0: break
            #molecule_tokenizer.create_input_from_list(new_blocks)

            messages = [
                {"role": "user", "content": cleaned_text},
            ]
            message_tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
            #print(message_tokens)

            frags = [[molecule_tokenizer.get_Idx(m) for m in mol.split('^')] for mol in mol_list]
            
            message_tokens = torch.Tensor(insert_sublists(message_tokens.squeeze(), frags, MOL_start, MOL_end)+[MOL_start]).to(torch.int).to(device)
            print(message_tokens, message_tokens.shape)
            
            generated = model.generate(
                input_ids=message_tokens.unsqueeze(0),
                max_new_tokens=32,
                num_beams=1,
                do_sample=False,
                #bad_words_ids=bad_words_ids,
            )
            #outputs = model.generate(message_tokens, max_new_tokens=128) 
            #out_text = tokenizer.decode(outputs[0])
            
            #print('Generated:', generated)

            #extracted_mols = extract_between_MOL(generated[0])
            #print(extracted_mols)
            #extracted_mols = [[molecule_tokenizer.get_block(e) for e in em] for em in extracted_mols]
            #print(extracted_mols)

            message_ids = generated[0, len(message_tokens):]
            print(message_ids)

            extracted_mols = message_ids > MOL_end
            locs = extracted_mols.nonzero().squeeze()
            #print(locs)

            #print(tokenizer.decode(generated[0].tolist()[len(message_tokens):], skip_special_tokens=True))

            tokens = tokenizer.convert_ids_to_tokens(message_ids, skip_special_tokens=True)
            #print(tokens)
            tokens = [molecule_tokenizer.get_block(int(message_ids[i]))+'^' if i in locs else t for i, t in enumerate(tokens)]
            #print(tokens)

            print(tokens)
            print()

            message = tokenizer.convert_tokens_to_string(tokens)
            print(message)
            print()
            
            mol_list, message = extract_mol_content(message)
            mol_list = [m[:-1] if m[-1]=='^' else m for m in mol_list]
            mol_list = [Chem.MolToSmiles(join_fragments(smi)) for smi in mol_list]

            #print(mol_list)
            
            message = replace(message, mol_list)

            print(message)
        #except Exception as e:
        #    print(e)


        