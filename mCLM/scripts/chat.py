import argparse
import os
import os.path as osp
import lightning as L
import sys
import io
import torch
from lightning import Trainer, seed_everything, Callback
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from mCLM.tokenizer.molecule_tokenizer import MoleculeTokenizer
from mCLM.data.processing import smiles_to_data, smiles_list_to_mols, extract_mol_content

from mCLM_tokenizer.tokenizer import convert_SMILES_strings

from transformers import AutoTokenizer

import pandas as pd
import pickle

from mCLM.data.dataloaders import KinaseDataModule
from mCLM.model.models import (
    mCLM,
)

import subprocess

from mCLM.data.processing import insert_sublists, find_first_occurrence


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
    parser.add_argument(
        "--pretrained_text_model",
        default="michiyasunaga/BioLinkBERT-base",
        type=str,
        help="Which text encoder to use from HuggingFace",
    )
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
    parser.add_argument("--ckpt_path", default="ckpts/test/", type=str)
    parser.add_argument("--ckpt", default="best_val_checkpoint.ckpt", type=str)
    parser.add_argument("--loss", default="CLIP", type=str)
    parser.add_argument("--load_ckpt", default=None, type=str)
    parser.add_argument("--load_GNN_ckpt", default=None, type=str)

    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--model", default="mCLM", type=str)
    parser.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct", type=str)
    #parser.add_argument(
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

    #with open(config["ckpt_path"] + "molecule_tokenizer.pkl", "rb") as f:
    #    molecule_tokenizer = pickle.load(f)
    #    molecule_tokenizer = torch.load(io.BytesIO(f.read()), map_location=torch.device('cpu'))

    with open(config["ckpt_path"] + "molecule_tokenizer.pth", "rb") as f:
        molecule_tokenizer = torch.load(f, map_location=torch.device('cpu'))

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

    model.load_state_dict(torch.load(config["ckpt_path"] + config['ckpt'], map_location='cpu')['state_dict'])

    print('Model Loaded')


    GNN_input_map = molecule_tokenizer.GNN_input_map
    
    #create a dictionary with a version of GNN_input_map for each device (the device is the key)
    for key in GNN_input_map:
        molecule_tokenizer.GNN_input_map[key] = GNN_input_map[key].to(model.device)

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
            {"role": "system", "content": "You are an expert chemist who designs molecules in a modular fashion or answers questions following the given instructions.",},
            {"role": "user", "content": "Please tell me a fact about a kinase inhibitor."},
            {"role": "assistant", "content": cleaned_text},
        ]
        message_tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        #print(message_tokens)

        frags = [[molecule_tokenizer.get_Idx(m) for m in mol.split('^')] for mol in mol_list]

        message_tokens = torch.Tensor(insert_sublists(message_tokens.squeeze(), frags, MOL_start, MOL_end)).to(torch.int)
        #print(message_tokens)

        outputs = model.generate(message_tokens, max_new_tokens=128) 
        out_text = tokenizer.decode(outputs[0])

        print(out_text)



