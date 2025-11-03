import argparse
import os
import os.path as osp
import lightning as L
import copy
import sys
import io
import re
from mCLM.tokenizer.utils import convert_instruction_to_input, message_ids_to_string, get_processor
import torch
from torch import nn
from lightning import Trainer, seed_everything, Callback
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger


from mCLM.tokenizer.molecule_tokenizer import MoleculeTokenizer


from transformers import AutoTokenizer


import pandas as pd
import pickle

from tqdm import tqdm

from mCLM.model.models import mCLM

from rdkit import Chem

import subprocess

from mCLM.data.processing import insert_sublists, find_first_occurrence, load_with_tqdm



if __name__ == "__main__":
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(description="mCLM")
    parser.add_argument("--trunc_length", default=512, type=int)

    parser.add_argument("--node_dim", default=138, type=int)
    parser.add_argument("--edge_dim", default=12, type=int)
    parser.add_argument("--hidden_dim_graph", default=512, type=int)
    parser.add_argument("--num_mp_layers", default=5, type=int)
    parser.add_argument("--num_readout_layers", default=1, type=int)
    parser.add_argument("--dropout", default=0.13, type=float)
    parser.add_argument("--aggr", default="mean", type=str)
    parser.add_argument("--jk", default="cat", type=str)

    parser.add_argument("--latent_size", default=256, type=int)

    parser.add_argument("--tokenizer_path", default="mCLM/OnlyBlocks/Top1k/Qwen2.5-3B_TotalTop1k_splitLoss_1kPretrain/", type=str)
    parser.add_argument("--ckpt_path", default="mCLM/OnlyBlocks/Top1k/Qwen2.5-3B_TotalTop1k_splitLoss_1kPretrain/", type=str)
    parser.add_argument("--ckpt", default="latest_checkpoint-epoch=04-step=129000.ckpt", type=str)

    
    parser.add_argument("--load_ckpt", default=None, type=str)
    parser.add_argument("--load_GNN_ckpt", default='mCLM_MolCLR/ckpts/OnlyBlocks/128_dim/', type=str)
    parser.add_argument("--pretrained_embeddings", default="mCLM_MolCLR/preprocess/Top500/128_dim/", type=str)
    parser.add_argument("--GNN_cache", default="", type=str)


    parser.add_argument("--model", default="mCLM", type=str)

    parser.add_argument("--pretrained_tokenizer", default="LLMs/Qwen2.5-3B/", type=str)

    parser.add_argument("--PEFT", type=bool, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    config = vars(args)

    print('Imports Done')

    device = 'cpu' #torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    if not 'molecule_tokenizer' in locals():
        tokenizer = AutoTokenizer.from_pretrained(config["pretrained_tokenizer"])
        tokenizer.pad_token = tokenizer.eos_token 
        tokenizer.add_tokens(['[MOL]', '[/MOL]'])

        torch.serialization.add_safe_globals([MoleculeTokenizer])
        molecule_tokenizer = load_with_tqdm(config["tokenizer_path"] + "molecule_tokenizer.pth", map_location=torch.device('cpu'), weights_only=False)#torch.load(f)
        
        start_idx = len(tokenizer)
        molecule_tokenizer.change_start_idx(start_idx)
        molecule_tokenizer.bfloat16 = True

        GNN_cache_path = config["tokenizer_path"] + 'molecule_tokenizer.graphs.pth'

        #Preprocess molecule tokenizer
        if osp.exists(GNN_cache_path):
            molecule_tokenizer.GNN_input_map = load_with_tqdm(GNN_cache_path, map_location=torch.device('cpu'), weights_only=False)
        else:
            molecule_tokenizer.create_input()
            with open(GNN_cache_path, "wb") as f:
                torch.save(molecule_tokenizer.GNN_input_map, f)


        MOL_start = tokenizer.convert_tokens_to_ids('[MOL]')
        MOL_end = tokenizer.convert_tokens_to_ids('[/MOL]')

        print('Tokenizer Loaded')

    if not 'model' in locals():

        model = mCLM(config)


        if config['pretrained_embeddings'] != None:
            loaded_pretrain_mol_embeddings = torch.load(config['pretrained_embeddings'] + 'precomputed_tokens.pt').to(torch.bfloat16)
            pretrain_mol_embeddings = nn.Embedding.from_pretrained(loaded_pretrain_mol_embeddings, freeze=True)
            pretrain_mol_embeddings.weight.data = pretrain_mol_embeddings.weight.data.to(torch.bfloat16)

            
            model.model.finalize_molecule_embeddings(embeddings=pretrain_mol_embeddings)
            model.model.use_mol_embeddings(True)
            model.pretrained_mol_embeddings = loaded_pretrain_mol_embeddings

        
        if config["load_GNN_ckpt"] != None:
            gnn_ckpt_sd = torch.load(config["load_GNN_ckpt"] + 'best_val_checkpoint.ckpt', map_location='cpu', weights_only=False)['state_dict']
            for key in list(gnn_ckpt_sd.keys()):
                gnn_ckpt_sd[key.replace('mol_encoder.', '')] = gnn_ckpt_sd.pop(key)

            model.model.model.model.mol_gnn.load_state_dict(gnn_ckpt_sd)
            print('Loaded GNN Weights')

        model.model.set_mol_vocab(molecule_tokenizer.GNN_input_map)
        model.model.extend_text_vocab_size(len(tokenizer.vocab))

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

        model.train(False)

        print('Model Set to Inference')

    bad_words_ids = None

        
    while True:
        user_input = input("Enter an instruction (type 'quit' to exit): ")
        if user_input == 'quit': break


        message_tokens = convert_instruction_to_input(user_input, model, molecule_tokenizer, tokenizer)
            
        ################## Generate results ###################################

        beam_size = 5            

        input_ids = message_tokens.to(device)

        processor = get_processor(molecule_tokenizer, tokenizer) #we do this every time in case vocab was expanded

        generated = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids), #This is to turn off the attention mask warning
            pad_token_id=tokenizer.eos_token_id, #This is to turn off the pad token warning
            max_new_tokens=32,
            num_beams=beam_size,
            num_return_sequences=beam_size,
            logits_processor=processor,
            do_sample=False,
            bad_words_ids=bad_words_ids,
            diversity_penalty=1.0,
            num_beam_groups=beam_size,
        )

        for i in [0]: #range(beam_size):
            message_ids = generated[i, message_tokens.shape[1]:]
            
            mol_msg, smiles_msg, smiles_list = message_ids_to_string(message_ids, molecule_tokenizer, tokenizer)

            if smiles_msg != None:
                print(mol_msg)
                if len(smiles_list) > 0:
                    print("SMILES list:", smiles_list)
            else:
                print(mol_msg)

            print()
            

        