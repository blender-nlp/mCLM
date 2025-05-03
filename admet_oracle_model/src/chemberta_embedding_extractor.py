import os
import torch
from tqdm import tqdm
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer

import warnings
warnings.filterwarnings("ignore")

from config import task, batch_size, ckpt_path, device

input_size = 384 # Input size for the GRU
hidden_size = 256 # Hidden size for the GRU
num_layers = 1 # Number of layers in the GRU
sequence_length = 512 # Sequence length for input data

class SequenceDataset(Dataset):
    def __init__(self, X):
        self.X = X
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx]

# Collate function for padding sequences in a batch
def collate_fn(X_batch):
    # Pad sequences to have the same length
    X_batch_padded = pad_sequence(X_batch, batch_first=True)
    return X_batch_padded

# GRU Classifier Model
class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(GRUClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.fc0 = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc0(x)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(out, c0)  # out: (batch_size, seq_length, hidden_size)
        out = out[:, -1, :]  # Get last hidden state
        out = self.relu(out)
        return out
    
def chemberta_embedding_extractor(SMILES_test, ckpt_path):
    
    ckpt_path = os.path.join(ckpt_path, f'{task}_chemberta.pt')

    tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-5M-MTR")
    model = AutoModel.from_pretrained("DeepChem/ChemBERTa-5M-MTR").to(device)

    X_test = []
    for sm in SMILES_test:#tqdm(SMILES_test, desc='ChemBERTa-2 feature extracting ...'):
        inputs = tokenizer(sm, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden_states = outputs.hidden_states[-1][0]
            X_test.append(last_hidden_states)

    test_dataset = SequenceDataset(X_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = GRUClassifier(input_size, hidden_size, num_layers).to(device)
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()  # Set model to evaluation mode
    
    test_features = []
    with torch.no_grad():  # No gradient computation for validation
        for X_test_batch in test_dataloader:
            X_test_batch = X_test_batch.to(device)
            test_outputs = model(X_test_batch).squeeze()
            test_outputs = test_outputs.cpu().detach().tolist()
            for output in test_outputs:
                test_features.append(output)

    return test_features
