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

from mCLM_tokenizer.tokenizer import convert_SMILES_strings, join_fragments, get_blocks

from transformers import AutoTokenizer
from transformers import LogitsProcessorList, LogitsProcessor, AutoTokenizer, AutoModelForCausalLM

import pandas as pd
import pickle

from tqdm import tqdm

from mCLM.data.dataloaders import KinaseDataModule
from mCLM.model.models import (
    mCLM,
)


# Add the src directory to sys.path
#sys.path.append(os.path.abspath('admet_oracle_model/src'))

#from admet_oracle_model.src.main import prepare_dataset, evaluate, MLP


from rdkit import Chem

import subprocess

from mCLM.data.processing import insert_sublists, find_first_occurrence

import json


def canonicalize(smi):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi))


def oracle_score(data_path, task):
    ckpt_path = 'admet_oracle_model/checkpoints'

    task_ckpt_path = os.path.join(ckpt_path, f'{task}_mlp.pt')

    dataloader, smiles = prepare_dataset(data_path, ckpt_path)
    model = MLP().to(device)

    model.load_state_dict(torch.load(task_ckpt_path))
    all_preds = evaluate(model, dataloader, device).squeeze()

    return all_preds


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
        diblock_mask = torch.BoolTensor([j in self.allowed_diblock_ids for j in range(scores.shape[1])])

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

            if input_ids[i, -1] in self.allowed_monoblock_ids and relative_pos != 1:
                scores[i, :] = float("-inf")
                scores[i, self.mol_end_token_id] = 0.0

            if input_ids[i, -1] in self.allowed_monoblock_ids and relative_pos == 1:
                scores[i, self.mol_end_token_id] = float("-inf")
                scores[i, ~diblock_mask] = float("-inf")

            if input_ids[i, -1] == self.mol_end_token_id:
                mask = torch.ones_like(scores[i,:], dtype=bool)
                mask[tokenizer.pad_token_id] = False
                #mask[self.mol_token_id] = False
                scores[i, ~mask] = float("-inf")
                scores[i, tokenizer.pad_token_id] = 0.0


        return scores



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

    parser.add_argument("--tokenizer_path", default="/shared/nas2/shared/llms/mCLM/OnlyBlocks/Top50V2/Qwen2.5-3B_TotalTop1k_splitLoss_1kPretrain/", type=str)
    #parser.add_argument("--ckpt_path", default="/shared/nas2/shared/llms/mCLM/OnlyBlocks/Top50V2/Qwen2.5-3B_TotalTop1k_splitLoss_1kPretrain/", type=str)
    #parser.add_argument("--ckpt", default="latest_checkpoint-epoch=04-step=129000.ckpt", type=str)
    parser.add_argument("--ckpt_path", default="/shared/nas2/shared/llms/mCLM/OnlyBlocks/Top50V2/Qwen2.5-3B_TotalTop1k_splitLoss_FinetuneV2/posneg/", type=str)
    parser.add_argument("--ckpt", default="best_val_checkpoint.ckpt", type=str)

    
    parser.add_argument("--loss", default="CLIP", type=str)
    parser.add_argument("--load_ckpt", default=None, type=str)
    parser.add_argument("--load_GNN_ckpt", default='/shared/nas2/cne2/Chemistry/mCLM_MolCLR/ckpts/OnlyBlocks/128_dim/', type=str)
    parser.add_argument("--pretrained_embeddings", default="/shared/nas2/cne2/Chemistry/mCLM_MolCLR/preprocess/Top500/128_dim/", type=str)
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
            loaded_pretrain_mol_embeddings = torch.load(config['pretrained_embeddings'] + 'precomputed_tokens.pt').to(torch.bfloat16)
            pretrain_mol_embeddings = nn.Embedding.from_pretrained(loaded_pretrain_mol_embeddings, freeze=True)
            pretrain_mol_embeddings.weight.data = pretrain_mol_embeddings.weight.data.to(torch.bfloat16)

            
            model.model.finalize_molecule_embeddings(embeddings=pretrain_mol_embeddings)
            #model.model.use_mol_embeddings(True)

        
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


    library = set()
    for block in molecule_tokenizer.block_to_idx:
        library.add(block)

    di_blocks = set([b for b in library if '1*' in b and '2*' in b])
    mono_blocks = set([b for b in library if ('3*' in b)])

    total_blocks = list(mono_blocks.union(di_blocks))


    def get_molecule(instruct, response, allowed_token_ids, num_beams=5):
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
            
        
        input_ids = message_tokens.long().unsqueeze(0)

        generated = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids), #This is to turn off the attention mask warning
            pad_token_id=tokenizer.eos_token_id, #This is to turn off the pad token warning
            max_new_tokens=10,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            do_sample=False,
            bad_words_ids=bad_words_ids,
            logits_processor=processor,
            num_beam_groups=num_beams, diversity_penalty=1.0,
            #no_repeat_ngram_size=2,
        )
        
        all_mol_list = []

        for j in range(num_beams):
            message_ids = generated[j, len(message_tokens):] 
            #print(message_ids)
            tokens = tokenizer.convert_ids_to_tokens(message_ids, skip_special_tokens=True)
            tokens = [molecule_tokenizer.get_block(int(message_ids[i]))+'^' if t == None else t for i, t in enumerate(tokens)]
            #print(tokens)
            #zz
            mol_msg, smiles_msg, mol_list = message_ids_to_string(message_ids, add_mol=True)

            #print(mol_msg)
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
    

    #while True:
    user_input = input("Enter a molecule (type 'quit' to exit): ")
    #if user_input == 'quit': break

    opt_tree = {}

    ############################## Add new blocks to the tokenizer ########################
    mols_list, MOL_input = convert_SMILES_strings('<SMILES>'+user_input+'</SMILES>')

    mol_list, cleaned_text = extract_mol_content(MOL_input)

    new_blocks = []

    for mol in mol_list:
        for m in mol.split('^'):
            new_blocks.append(m)
            #molecule_tokenizer.add_block(m)
    new_blocks = [nb for nb in new_blocks if nb not in molecule_tokenizer.block_to_idx]
    print('New Blocks:', len(new_blocks))
    if len(new_blocks) > 0: 
        for nb in new_blocks: molecule_tokenizer.add_block(nb)
        molecule_tokenizer.create_input_from_list(new_blocks)
        model.model.set_mol_vocab(molecule_tokenizer.GNN_input_map)

        embs = []
        for nb in new_blocks:
            graph = molecule_tokenizer.get(molecule_tokenizer.get_Idx(nb))
            emb = model.model.model.model.mol_gnn(graph)
            embs.append(emb)
        embs = torch.cat(embs, dim=0)

        loaded_pretrain_mol_embeddings = torch.cat([loaded_pretrain_mol_embeddings, embs], dim=0)

        pretrain_mol_embeddings = nn.Embedding.from_pretrained(loaded_pretrain_mol_embeddings, freeze=True)
        pretrain_mol_embeddings.weight.data = pretrain_mol_embeddings.weight.data.to(torch.bfloat16)

        model.model.finalize_molecule_embeddings(embeddings=pretrain_mol_embeddings)
        model.model.use_mol_embeddings(True)
    ################################################################

    def run_opt(MOL_input, task, num_beams=10):

        #Uncomment this return statement to test logic
        #return list(zip([MOL_input+'A', MOL_input+'B'], [MOL_input+'D', MOL_input+'E']))

        MOL_input = templates[task].replace('[MOL] [/MOL]', MOL_input)

        token_ids, mols = get_molecule(MOL_input, '', allowed_token_ids=list(molecule_tokenizer.idx_to_block.keys()), num_beams=num_beams)

        start_smiles = Chem.MolToSmiles(join_fragments(mol_list[0]))

        #Ignore duplicates
        mols = list(set(mols))


        if len(mols) != 0:
            new_smis = []
            for mol in mols:
                new_smi = Chem.MolToSmiles(join_fragments(mol))#'.'.join([Chem.MolToSmiles(join_fragments(mol)) for mol in inp])

                new_smis.append(new_smi)


            #print('Original:\t', start_smiles, mol_list[0])
            #for smi, mol in zip(new_smis,  mols):
            #    print('Generated:\t', smi, mol)

            return list(zip(new_smis, mols))

        else:
            #print(task,smi, blck, 'failed')
            #zz
            return list(zip([], []))

    result = run_opt(MOL_input, 'ames')

    #print(result)
    #zz

    keys = list(templates.keys())


    import json
    from tqdm import tqdm

    def build_opt_tree(input_value, keys, depth, run_opt, save_path="opt_tree_save.json", save_every=10, num_beams=10):
        cache = {}  # memoization
        total_nodes = sum(len(keys)**i for i in range(1, depth + 1))
        pbar = tqdm(total=total_nodes, desc="Building opt_tree")
        step = [0]
        opt_tree_ref = [{}]  # mutable container to hold the root reference

        def _run_opt_cached(input_val, key):
            
            cache_key = json.dumps((input_val, key), sort_keys=True)
            if cache_key not in cache:
                cache[cache_key] = run_opt(input_val[1], key, num_beams=num_beams)
            return cache[cache_key]

        def _build(input_val, current_depth):
            if current_depth == 0:
                return input_val

            node = {}
            for key in keys:
                result = _run_opt_cached(input_val, key)
                pbar.update(1)
                pbar.set_description(f'Unique Molecules: {len(cache)}')
                node[key] = _build(result[1], current_depth - 1)

                # Periodically save the root tree
                step[0] += 1
                if step[0] % save_every == 0:
                    with open(save_path, "w") as f:
                        json.dump(opt_tree_ref[0], f, indent=2)

            return (input_val, node)

        # Build and assign to reference
        opt_tree_ref[0] = _build(input_value, depth)

        # Final save
        with open(save_path, "w") as f:
            json.dump(opt_tree_ref[0], f, indent=2)
        pbar.close()

        return opt_tree_ref[0], cache



    DEPTH = 2
    NUM_BEAMS = 3


    # Example usage:
    opt_tree, _ = build_opt_tree(input_value=(user_input, MOL_input), keys=keys, depth=DEPTH, run_opt=run_opt, num_beams=NUM_BEAMS)

    print(opt_tree)

    with open(f'{user_input}.OptTree.json', 'w') as f:
        json.dump(opt_tree, f, indent=4)


