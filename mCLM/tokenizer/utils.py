
import torch

from transformers import LogitsProcessorList, LogitsProcessor, AutoTokenizer, AutoModelForCausalLM

from torch import nn

import copy
import re

from rdkit import Chem

from mCLM_tokenizer.tokenizer import convert_SMILES_strings, join_fragments
from mCLM.data.processing import smiles_to_data, smiles_list_to_mols, extract_mol_content
from mCLM.data.processing import insert_sublists, find_first_occurrence

def extract_between_MOL(tensor, tokenizer):
    tensor_list = tensor.tolist()  # Convert tensor to list
    extracted = []
    temp = []
    recording = False  # Flag to start recording elements


    MOL_start = tokenizer.convert_tokens_to_ids('[MOL]')
    MOL_end = tokenizer.convert_tokens_to_ids('[/MOL]')

    for num in tensor_list:
        if num == MOL_start or num == MOL_end:
            if recording and temp:
                extracted.append(temp)  # Save previous group
            temp = []  # Reset temp list
            recording = True  # Start recording
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


def message_ids_to_string(message_ids, molecule_tokenizer, tokenizer, add_mol=False):
    
    MOL_start = tokenizer.convert_tokens_to_ids('[MOL]')
    MOL_end = tokenizer.convert_tokens_to_ids('[/MOL]')

    extracted_mols = message_ids > MOL_end
    locs = extracted_mols.nonzero().squeeze()

    tokens = tokenizer.convert_ids_to_tokens(message_ids, skip_special_tokens=False)

    tokens = [molecule_tokenizer.get_block(int(message_ids[i]))+'^' if i in locs else t for i, t in enumerate(tokens)]

    clean_tokens = [tok for tok in tokens if tok not in tokenizer.all_special_tokens]


    message = '[MOL]'*add_mol + tokenizer.convert_tokens_to_string(clean_tokens)

    
    try:
        mol_list, smi_message = extract_mol_content(message)
        mol_list = [m[:-1] if m[-1]=='^' else m for m in mol_list]
        smiles_list = [Chem.MolToSmiles(join_fragments(smi)) for smi in mol_list]                
        smi_message = replace(smi_message, mol_list)
    except Exception as e:
        smi_message = None
        smiles_list = None

    return message, smi_message, mol_list, smiles_list


def convert_instruction_to_input(instruction, model, molecule_tokenizer, tokenizer):
    
    MOL_start = tokenizer.convert_tokens_to_ids('[MOL]')
    MOL_end = tokenizer.convert_tokens_to_ids('[/MOL]')

    mols_list, MOL_input = convert_SMILES_strings(instruction)

    mol_list, cleaned_text = extract_mol_content(MOL_input)


    ################## Create new vocabulary ##########################
    new_blocks = []

    for mol in mol_list:
        for m in mol.split('^'):
            new_blocks.append(m)
            
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

        model.pretrained_mol_embeddings = torch.cat([model.pretrained_mol_embeddings, embs], dim=0)

        pretrain_mol_embeddings = nn.Embedding.from_pretrained(model.pretrained_mol_embeddings, freeze=True)
        pretrain_mol_embeddings.weight.data = pretrain_mol_embeddings.weight.data.to(torch.bfloat16)

        model.model.finalize_molecule_embeddings(embeddings=pretrain_mol_embeddings)
        

    ################# Create messages with molecule tokens ########################
    messages = [
        {"role": "user", "content": cleaned_text},
    ]
    message_tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")

    frags = [[molecule_tokenizer.get_Idx(m) for m in mol.split('^')] for mol in mol_list]
    
    message_tokens = torch.Tensor(insert_sublists(message_tokens.squeeze(), frags, MOL_start, MOL_end)).to(torch.long).unsqueeze(0)

    return message_tokens




class ConditionalMolEndProcessor(LogitsProcessor):
    def __init__(self, mol_token_id: int, mol_end_token_id: int, end_token: int, allowed_monoblock_ids: set, allowed_diblock_ids: set):
        self.mol_token_id = mol_token_id
        self.mol_end_token_id = mol_end_token_id
        self.end_token = end_token
        self.allowed_monoblock_ids = allowed_monoblock_ids
        self.allowed_diblock_ids = allowed_diblock_ids
        self.allowed_block_ids = allowed_diblock_ids | allowed_monoblock_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # input_ids shape: (batch_size, sequence_length)
        batch_size, seq_len = input_ids.shape
        monoblock_mask = torch.BoolTensor([j in self.allowed_monoblock_ids for j in range(scores.shape[1])]) # or j == self.mol_end_token_id
        diblock_mask = torch.BoolTensor([j in self.allowed_diblock_ids for j in range(scores.shape[1])])
        #MOL_mask = torch.BoolTensor([j == self.mol_token_id or j == self.mol_end_token_id for j in range(scores.shape[1])])

        block_mask = monoblock_mask | diblock_mask

        for i in range(batch_size):

            last_id = input_ids[i, -1].item()


            #Don't generate blocks randomly
            if (last_id not in self.allowed_block_ids and (last_id != self.mol_token_id)):# or input_ids[i, -1] == self.mol_end_token_id:
                scores[i, block_mask] = float("-inf")
                scores[i, self.mol_end_token_id] = float("-inf")
                continue

            #Don't generate MOL_end randomly
            if last_id not in self.allowed_block_ids:
                scores[i, self.mol_end_token_id] = float("-inf")


            input_seq = input_ids[i]
            # Find the last occurrence of [MOL] in the sequence
            mol_positions = (input_seq == self.mol_token_id).nonzero(as_tuple=True)[0]
            if len(mol_positions) == 0:
                continue  # [MOL] not found, do nothing
            mol_start = mol_positions[-1].item()  # start counting after last [MOL]
            relative_pos = seq_len - mol_start - 1  # how many tokens after [MOL]

            #First we have to generate a monoblock
            if relative_pos == 0:
                scores[i, ~monoblock_mask] = float("-inf")

            #We can't finish a molecule after a diblock
            if last_id in self.allowed_diblock_ids:
                scores[i, self.mol_end_token_id] = float("-inf")

            # After a second monoblock is generated, we have to finish the molecule
            if last_id in self.allowed_monoblock_ids and relative_pos != 1:
                scores[i, :] = float("-inf")
                scores[i, self.mol_end_token_id] = 0.0

            # Max molecule length is 5
            if last_id in self.allowed_diblock_ids and relative_pos > 3:
                scores[i, ~monoblock_mask] = float("-inf")

            #Force molecule to be >= 2 blocks
            if last_id in self.allowed_monoblock_ids and relative_pos == 1:
                #scores[i, self.mol_end_token_id] = float("-inf") #Force >2 blocks
                scores[i, ~block_mask] = float("-inf") 

            #Force the model to stop talking after generating a molecule
            #if last_id == self.mol_end_token_id:
            #    scores[i, :] = float("-inf")
            #    scores[i, tokenizer.pad_token_id] = 0.0


        return scores

def get_processor(molecule_tokenizer, tokenizer):
    
    MOL_start = tokenizer.convert_tokens_to_ids('[MOL]')
    MOL_end = tokenizer.convert_tokens_to_ids('[/MOL]')
    
    library = set()
    for block in molecule_tokenizer.block_to_idx:
        library.add(block)

    di_blocks = set([b for b in library if '1*' in b and '2*' in b])
    mono_blocks = set([b for b in library if ('3*' in b)])

    total_blocks = list(mono_blocks.union(di_blocks))

    processor = LogitsProcessorList([ConditionalMolEndProcessor(MOL_start, MOL_end, tokenizer.convert_tokens_to_ids(tokenizer.eos_token), \
            set([molecule_tokenizer.get_Idx(b) for b in mono_blocks]), \
            set([molecule_tokenizer.get_Idx(b) for b in di_blocks])), \
        ])
    return processor
    