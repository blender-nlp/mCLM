
from rdkit import Chem

import os
import io
import json

import re

import torch

import pandas as pd

from collections import defaultdict

from tqdm import tqdm


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


molecule_tokenizer = load_with_tqdm("../GNN_input_cache/Total.molecule_tokenizer.v2.pth", map_location=torch.device('cpu'), weights_only=False)#torch.load(f)

total_blocks = set(molecule_tokenizer.block_to_idx.keys()) - set([''])

def write_strings_to_file(strings: set, file_path: str):
    with open(file_path, "w") as f:
        for s in strings:
            f.write(s + "\n")

write_strings_to_file(total_blocks, 'total_blocks.txt')

