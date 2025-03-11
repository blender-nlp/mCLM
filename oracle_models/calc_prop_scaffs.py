import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt

from collections import defaultdict


# Load the dataset
df = pd.read_csv("data_scaffolds.csv", index_col='molecule')


props = pd.Series(list((set(df.keys()) - set(['molecule', 'scaffolds']))))


smi_to_scaffold = dict(zip(df.index, df['scaffolds']))

scaff_groups = defaultdict(set)
for smi in smi_to_scaffold: scaff_groups[smi_to_scaffold[smi]].add(smi)

new_props = []

avg_sim_mols = []
sum_sim_mols = []

pos_neg_pairs = set()

bar = tqdm(props)
for p in bar:
    pos_mols = set(df[df[p] == 1].index)

    if len(pos_mols) == 0: continue

    sim_mol_count = []

    save_len = {} #DP for speedup

    for pmol in pos_mols:#tqdm(pos_mols, desc='Property Molecules'):
        scaff = df['scaffolds'][pmol]
        if scaff in save_len: 
            sim_mol_count.append(save_len[scaff])
        else:
            #sim_mols = scaff_groups[scaff]
            sim_mols = scaff_groups[scaff] - pos_mols #[sm for sm in sim_mols if sm not in pos_mols]
            #if len(sim_mols) > 0:
            #    print(p, pmol, list(sim_mols)[0], scaff)
            sim_mol_count.append(len(sim_mols))
            save_len[scaff] = len(sim_mols)
        for s in scaff_groups[scaff]: pos_neg_pairs.add((pmol, s))


    #print(p + ":", len(pos_mols), "molecules.\n\tAverage Similar Negatives:", np.mean(sim_mol_count))

    avg_sim_mols.append(np.mean(sim_mol_count))
    sum_sim_mols.append(np.sum(sim_mol_count))
    new_props.append(p)

    bar.set_description(p + ": " + str(len(pos_mols)) + " molecules. " + str(np.mean(sim_mol_count)) + " avg. negatives.")

new_props = pd.Series(new_props)

#bins = int(max(sum_sim_mols))#100  # Number of bins

lin_to_log = 100
bins = np.concatenate((np.linspace(0, lin_to_log, num=100), np.logspace(np.log10(lin_to_log), np.log10(max(sum_sim_mols)), num=10)))


# Compute histograms
hist_avg, bin_edges_avg = np.histogram(avg_sim_mols, bins=bins)
hist_sum, bin_edges_sum = np.histogram(sum_sim_mols, bins=bins)

# Normalize sum_sim_mols to match avg_sim_mols scale
scaling_factor = max(hist_avg) / max(hist_sum)
hist_sum_scaled = hist_sum * scaling_factor

# Compute bin centers
bin_centers_avg = (bin_edges_avg[:-1] + bin_edges_avg[1:]) / 2
bin_centers_sum = (bin_edges_sum[:-1] + bin_edges_sum[1:]) / 2

fig, ax = plt.subplots()

# Plot avg_sim_mols histogram (bottom)
ax.bar(bin_centers_sum, hist_avg, width=np.diff(bin_edges_sum), align='center', edgecolor='black', alpha=0.7, label='avg_sim_mols')
ax.bar(bin_centers_sum, -hist_sum_scaled, width=np.diff(bin_edges_sum), align='center', edgecolor='black', alpha=0.7, label='sum_sim_mols')

# Set y-axis
ax.axhline(0, color='black', linewidth=1)
ax.set_ylabel('Frequency')
ax.set_ylim(-max(hist_sum_scaled) * 1.1, max(hist_avg) * 1.1)
plt.yscale('symlog', linthresh=1)  # Keeps values near 0 linear but applies log scaling

plt.xscale('symlog', linthresh=lin_to_log, linscale =4)
plt.xlim(0, max(sum_sim_mols))

# Set the lower x-axis for sum_sim_mols
ax.set_xlabel('sum_sim_mols')

# Create upper x-axis for avg_sim_mols
def tick_function(x):
    """ Convert sum_sim_mols bins to avg_sim_mols bins (approximate mapping) """
    return np.interp(x, bin_edges_sum, bin_edges_avg)  # Linear interpolation

ax_top = ax.secondary_xaxis('top', functions=(tick_function, tick_function))
ax_top.set_xlabel('avg_sim_mols')
ax_top.set_xscale('symlog', linthresh=lin_to_log, linscale =4)

#plt.legend()

plt.savefig('scaff_sim_histo.png')

plt.close()


print('Number of Properties with at least 10 Synthetic Data Points:', sum(np.array(sum_sim_mols) > 10))


