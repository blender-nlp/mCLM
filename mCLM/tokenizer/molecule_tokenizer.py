

from tqdm import tqdm

from mCLM.data.processing import smiles_to_data


class MoleculeTokenizer:
    def __init__(self, start_idx):
        """
        Initializes the MoleculeTokenizer.
        """
        self.start_idx = start_idx
        self.block_to_idx = {}
        self.GNN_input_map = {}


    def add_block(self, block: str):
        if block not in self.block_to_idx:
            self.block_to_idx[block] = self.start_idx + len(self.block_to_idx)


    def create_input(self):
        for block in self.block_to_idx: #tqdm(block_to_idx, desc='Creating GNN Input'):
            if self.block_to_idx[block] not in self.GNN_input_map:
                self.GNN_input_map[self.block_to_idx[block]] = smiles_to_data(block)


    def get_Idx(self, block: str):
        """
        Tokenizes the given block ID.

        Args:
            molecule (str): The molecular representation to tokenize.

        Returns:
            list: A list of tokens representing the molecule.
        """
        return self.block_to_idx[block]


    def get(self, ID: str):
        """
        Tokenizes the given block ID.

        Args:
            molecule (str): The molecular representation to tokenize.

        Returns:
            list: A list of tokens representing the molecule.
        """
        return self.GNN_input_map[ID]



