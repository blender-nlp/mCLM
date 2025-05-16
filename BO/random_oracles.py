import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import re
from rdkit import Chem
from rdkit.Chem import Crippen
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError

from collections import defaultdict

import os
import sys

from scipy.stats import hmean

from mCLM_tokenizer.tokenizer import join_fragments

device = 'cpu' #torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Add the src directory to sys.path
sys.path.append(os.path.abspath('../admet_oracle_model/src'))

from admet_oracle_model.src.main import prepare_dataset, evaluate, MLP, evaluate_computed

tasks = ['ames', 'bbbp', 'cyp3a4', 'dili', 'hia', 'pgp']

task_type = {
    'ames':'min',
    'bbbp':'max',
    'cyp3a4':'min',
    'dili':'min',
    'hia':'max',
    'pgp':'min',
}

ckpt_path = '../admet_oracle_model/checkpoints'

oracle_models = {}

for task in tasks:
    task_ckpt_path = os.path.join(ckpt_path, f'{task}_mlp.pt')

    oracle_models[task] = MLP().to(device)

    oracle_models[task].load_state_dict(torch.load(task_ckpt_path))



# Load block to tensor index mapping
map_info = pd.read_csv("map_info.txt")
block_to_idx = dict(zip(map_info['block'], map_info['tensor_idx']))

def select_blocks(map_info, n_caps=200, n_mids=100, seed=42):
    random.seed(seed)
    blocks = map_info['block'].tolist()

    cap_blocks = [b for b in blocks if '[3*]' in b and '[1*]' not in b and '[2*]' not in b]
    mid_blocks = [b for b in blocks if '[1*]' in b and '[2*]' in b]

    selected_caps = random.sample(cap_blocks, min(n_caps, len(cap_blocks)))
    selected_mids = random.sample(mid_blocks, min(n_mids, len(mid_blocks)))

    return selected_caps, selected_mids

valid_caps, valid_mids = select_blocks(map_info)
print(f"Selected {len(valid_caps)} caps and {len(valid_mids)} mids.")

# Load precomputed embeddings
embeddings = torch.load("precomputed_tokens.pt")

def get_embedding(smiles):
    idx = block_to_idx.get(smiles)
    if idx is None:
        raise ValueError(f"Block not found in map_info: {smiles}")
    return embeddings[idx].numpy()


def oracle_score(smiles):
    #preds = evaluate_computed(smiles) #actually a list of smiles
    preds = defaultdict(list)

    for task in tasks:
        model = oracle_models[task]

        dataloader, smiles = prepare_dataset(smiles, ckpt_path)

        all_preds = evaluate(model, dataloader, device).squeeze()

        if task_type[task] == 'min':
            all_preds = 1 - all_preds

        preds[task] = all_preds

    keys = list(preds.keys())
    for i, smi in enumerate(smiles):
        preds['hmean'].append(hmean([preds[key][i] for key in keys]))


    return preds


# Simulated LLM oracle
def simulated_llm_property(smiles):
    ors = oracle_score(smiles)

    for key in ors:
        for i in range(len(ors[key])):
            ors[key][i] += np.random.normal(0, 0.25)
    return ors

    #return Crippen.MolLogP(mol) + np.random.normal(0, 1.0)  # simulate some noise


    
# Embed triple (start, mid, end) into vector
def get_molecule_embedding(start, mid, end):
    try:
        s_emb = get_embedding(start)
        m_emb = get_embedding(mid)
        e_emb = get_embedding(end)
        return np.concatenate([s_emb, m_emb, e_emb])
    except Exception as e:
        print(f"[embedding error] {e}")
        return None

# Find nearest fragment by cosine similarity
def find_nearest_fragment(target_emb, fragment_list, is_mid=False):
    best_sim = -1
    best_frag = None
    for frag in fragment_list:
        try:
            frag_emb = get_embedding(frag)
            sim = np.dot(target_emb, frag_emb) / (np.linalg.norm(target_emb) * np.linalg.norm(frag_emb))
            if sim > best_sim:
                best_sim = sim
                best_frag = frag
        except:
            continue
    return best_frag

# Visualization of active learning loop
def plot_llm_vs_oracle(llm_r2_history, llm_mse_history):
    rounds = list(range(1, len(llm_r2_history)+1))
    fig, ax1 = plt.subplots(figsize=(8,5))

    ax1.set_xlabel("Round")
    ax1.set_ylabel("LLM R²", color='tab:blue')
    ax1.plot(rounds, llm_r2_history, marker='o', color='tab:blue', label="LLM R²")
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel("LLM MSE", color='tab:red')
    ax2.plot(rounds, llm_mse_history, marker='s', linestyle='--', color='tab:red', label="LLM MSE")
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title("LLM Prediction vs Oracle (per Round)")
    fig.tight_layout()
    plt.savefig('results_random.png')
    

def plot_cum_max_value(cum_max, llm_cum_max):
    rounds = list(range(1, len(cum_max)+1))
    fig = plt.figure()

    plt.xlabel("Round")
    plt.plot(rounds, cum_max, marker='o', color='tab:blue', label='Oracle')
    #plt.plot(rounds, llm_cum_max, marker='s', color='tab:orange', label='LLM')

    plt.title("Max Cumulative Score vs Oracle (per Round)")
    plt.legend()
    plt.tight_layout()
    plt.savefig('cum_max_value_random.png')



class SurrogateNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)


class ActiveLearner:
    def __init__(self, caps, mids, embedding_dim):
        self.caps = caps
        self.mids = mids
        self.X = []
        self.y = []

        self.scaler = StandardScaler()
        self.model = SurrogateNN(input_dim=embedding_dim * 3)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()


    def add_observation(self, emb, prop):
        self.X.append(emb)
        self.y.append(prop)

    def train(self):
        if len(self.X) < 10:
            return
        X_array = np.array(self.X)
        y_array = np.array(self.y).reshape(-1, 1)

        if len(self.X) > 10:
            max_components = min(len(self.X), X_array.shape[1])

        X_scaled = self.scaler.fit_transform(X_array)

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_array, dtype=torch.float32)

        self.model.train()
        for _ in range(200):  # epochs
            self.optimizer.zero_grad()
            y_pred = self.model(X_tensor)
            loss = self.criterion(y_pred, y_tensor)
            loss.backward()
            self.optimizer.step()

    def predict(self, emb):
        self.model.eval()
        try:
            emb_scaled = self.scaler.transform([emb])
        except NotFittedError:
            print("[predict] Skipping prediction: Scaler not fitted.")
            return None
    
        emb_tensor = torch.tensor(emb_scaled, dtype=torch.float32)
        with torch.no_grad():
            return self.model(emb_tensor).item()


    def suggest_top_uncertain(self, n=100):
        suggestions = []
        for _ in range(300):
            s = random.choice(self.caps)
            m = random.choice(self.mids)
            e = random.choice(self.caps)
            emb = get_molecule_embedding(s, m, e)
            if emb is None:
                continue
            
            # Simulate uncertainty by sampling predictions multiple times
            pred_samples = []
            for _ in range(10):  # Number of Monte Carlo samples for uncertainty estimation
                pred = self.predict(emb)
                if pred is not None:
                    pred_samples.append(pred)
            
            if len(pred_samples) < 5:
                continue
    
            # Calculate variance of the predictions
            pred_variance = np.var(pred_samples)
            
            # Use the variance as a proxy for uncertainty
            suggestions.append((s, m, e, pred_variance))  # Higher variance = more uncertain
        
        # Sort by uncertainty (variance) and select the top `n` uncertain molecules
        sorted_suggestions = sorted(suggestions, key=lambda x: x[3], reverse=True)
        return sorted_suggestions[:n]


    def get_metrics(self):
        if not self.X:
            return None, None
        X_array = np.array(self.X)
        y_array = np.array(self.y)

        X_scaled = self.scaler.transform(X_array)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_tensor).numpy().flatten()
        return r2_score(y_array, y_pred), mean_squared_error(y_array, y_pred)

# Main active learning loop
def active_learning_loop(n_rounds=10):
    learner = ActiveLearner(valid_caps, valid_mids, embedding_dim=embeddings[0].shape[0])
    all_results = []

    # Zero-shot: no initial training, start from suggestions
    print("[zero-shot] Starting cold...")
    mols = []
    for i in range(10): #while len(learner.X) < 10: # Random suggestions for starting
            s = random.choice(valid_caps)
            m = random.choice(valid_mids)
            e = random.choice(valid_caps)
            mol = join_fragments(f'{s}^{m}^{e}')
            if mol is None:
                continue
            emb = get_molecule_embedding(s, m, e)
            if emb is None:
                continue
            mols.append((s, m, e, Chem.MolToSmiles(mol), emb))

    oracle = oracle_score([smi for s, m, e, smi, _ in mols])
    preds = simulated_llm_property([smi for s, m, e, smi, _ in mols])

    #print(oracle['hmean'])
    #print(preds['hmean'])
    #zz

    for (s, m, e, smi, emb), y, o in zip(mols, preds['hmean'], oracle['hmean']):
        learner.add_observation(emb, o)
        all_results.append((s, m, e, Chem.MolToSmiles(mol), y, o))
    
    learner.train()
    llm_r2_history = []
    llm_mse_history = []
    cumulative_max = []
    LLM_cumulative_max = []
    r2, mse = learner.get_metrics()
    llm_preds = [entry[4] for entry in all_results]     # simulated LLM values
    oracle_vals = [entry[5] for entry in all_results]   # true LogP values
    
    cumulative_max.append(max(oracle_vals))
    LLM_cumulative_max.append(max(llm_preds))

    llm_r2 = r2_score(oracle_vals, llm_preds)
    llm_mse = mean_squared_error(oracle_vals, llm_preds)
    llm_r2_history.append(llm_r2)
    llm_mse_history.append(llm_mse)
    print(f" Model R²: {r2:.3f}, MSE: {mse:.3f} | LLM R²: {llm_r2:.3f}, LLM MSE: {llm_mse:.3f}")
    
    for r in range(n_rounds-1):
        print(f"\n[Round {r+1}] Suggesting molecules...")
        suggestions = learner.suggest_top_uncertain(100)
        if len(suggestions) < 10:
            print("[!] Not enough suggestions; skipping.")
            break
        #selected = random.sample(suggestions, 10)  # Simulate LLM selection
        

        mols = []
        for s, m, e, _ in suggestions:
            mol = join_fragments(f'{s}^{m}^{e}')
            if mol is None:
                continue
            emb = get_molecule_embedding(s, m, e)
            if emb is None:
                continue
            mols.append((s, m, e, Chem.MolToSmiles(mol), emb))
        
        selected = random.sample(mols, 10)  # Simulate LLM selection
        
        oracle = oracle_score([smi for s, m, e, smi, _ in selected])
        preds = simulated_llm_property([smi for s, m, e, smi, _ in selected])
        
        for (s, m, e, smi, emb), y, o in zip(selected, preds['hmean'], oracle['hmean']):
            learner.add_observation(emb, o)
            all_results.append((s, m, e, smi, y, o))
            

        learner.train()
        r2, mse = learner.get_metrics()
        llm_preds = [entry[4] for entry in all_results]     # simulated LLM values
        oracle_vals = [entry[5] for entry in all_results]   # true LogP values

        llm_r2 = r2_score(oracle_vals, llm_preds)
        llm_mse = mean_squared_error(oracle_vals, llm_preds)
        llm_r2_history.append(llm_r2)
        llm_mse_history.append(llm_mse)
        cumulative_max.append(max(oracle_vals))
        LLM_cumulative_max.append(max(llm_preds))
        
        print(f" Model R²: {r2:.3f}, MSE: {mse:.3f} | LLM R²: {llm_r2:.3f}, LLM MSE: {llm_mse:.3f} | Cumulative Max: {max(oracle_vals):.3f} | LLM Cumulative Max: {max(llm_preds):.3f}")

    plot_llm_vs_oracle(llm_r2_history, llm_mse_history)
    plot_cum_max_value(cumulative_max, LLM_cumulative_max)
    return all_results

# Run
results = active_learning_loop()



print(results)



