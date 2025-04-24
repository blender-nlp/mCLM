
from rdkit import Chem

from tqdm import tqdm

from mCLM.data.processing import smiles_to_data

import torch

def canonicalize(smi):
    try:
        rv = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    except:
        rv = None
    return rv

def convert_data_to_bfloat16(data):
    for key, item in data:
        if isinstance(item, torch.Tensor) and item.dtype.is_floating_point:
            data[key] = item.to(dtype=torch.bfloat16)
    return data

class MoleculeTokenizer:
    def __init__(self, start_idx, bfloat16=False):
        """
        Initializes the MoleculeTokenizer.
        """
        self.start_idx = start_idx
        self.block_to_idx = {}
        self.idx_to_block = {}
        self.GNN_input_map = {}
        self.bfloat16 = bfloat16

        self.bad_blocks = set()

    def __len__(self):
        return len(self.block_to_idx)

    def add_block(self, block: str):
        if block not in self.block_to_idx:
            #if canonicalize(block) == None: 
            #    #print('uncanonical block', block)
            #    self.bad_blocks.add(block)
            #    return

            self.block_to_idx[block] = self.start_idx + len(self.block_to_idx)
        if self.block_to_idx[block] not in self.idx_to_block:
            self.idx_to_block[self.block_to_idx[block]] = block

    def create_input(self):
        for block in tqdm(self.block_to_idx, desc='Creating GNN Input'):
            if self.block_to_idx[block] not in self.GNN_input_map:
                try:
                    self.GNN_input_map[self.block_to_idx[block]] = smiles_to_data(block)
                    if self.bfloat16:
                        convert_data_to_bfloat16(self.GNN_input_map[self.block_to_idx[block]])
                except Exception as e:
                    print(block, e)
                    self.bad_blocks.add(block)
                    #zz
                    

    def create_input_from_list(self, blocks):
        for block in blocks:
            if self.block_to_idx[block] not in self.GNN_input_map:
                try:
                    self.GNN_input_map[self.block_to_idx[block]] = smiles_to_data(block)
                    if self.bfloat16:
                        convert_data_to_bfloat16(self.GNN_input_map[self.block_to_idx[block]])
                except:
                    print('Bad Block:', block)
                    self.bad_blocks.add(block)
                    
                    self.GNN_input_map[self.block_to_idx[block]] = smiles_to_data('[1*][2*]') #UNK value, essentially 

                    if self.bfloat16:
                        convert_data_to_bfloat16(self.GNN_input_map[self.block_to_idx[block]])


    def clear_data(self): #free up RAM after validation
        print('Clearing molecule tokenizer data')
        self.GNN_input_map = {}


    #Allows switching between language models
    def change_start_idx(self, new_start_idx):
        old_start_idx = self.start_idx
        #print(f'in change_start_idx with new_start_idx={new_start_idx} and old_start_idx={old_start_idx}')
        old_new_map = {}
        self.idx_to_block = {}

        for block in tqdm(self.block_to_idx, desc='Changing start index'):
            old_idx = self.block_to_idx[block]
            new_idx = old_idx - old_start_idx + new_start_idx
            self.block_to_idx[block] = new_idx
            self.idx_to_block[new_idx] = block
            old_new_map[old_idx] = new_idx

        self.GNN_input_map = {old_new_map.get(k, k): v for k, v in self.GNN_input_map.items()}
        self.start_idx = new_start_idx


    def get_Idx(self, block: str):
        """
        Tokenizes the given block ID.

        Args:
            molecule (str): The molecular representation to tokenize.

        Returns:
            list: A list of tokens representing the molecule.
        """
        return self.block_to_idx[block]


    def get_block(self, ID: int):
        """
        Tokenizes the given block idx.

        Args:
            molecule (str): The molecular representation to tokenize.

        Returns:
            list: A list of tokens representing the molecule.
        """
        return self.idx_to_block[ID]


    def get(self, ID: int):
        """
        Tokenizes the given block ID.

        Args:
            molecule (str): The molecular representation to tokenize.

        Returns:
            list: A list of tokens representing the molecule.
        """
        try:
            if ID in self.GNN_input_map:
                return self.GNN_input_map[ID]
            else:
                self.create_input_from_list([self.get_block(ID)])
                return self.GNN_input_map[ID]
        except KeyError as e:
            print(f"KeyError: {e} not found in dictionary. Using UNK replacement.")
            return smiles_to_data('[1*][2*]')



