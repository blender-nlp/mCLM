

import pickle

from collections import defaultdict

from rdkit import Chem

from scipy.stats import ttest_rel, ttest_ind, mannwhitneyu
from scipy.stats import wilcoxon

import numpy as np

def bootstrap_mean_diff(a, b, n_iter=10000):
    observed = np.mean(a - b)
    diffs = []

    for _ in range(n_iter):
        idx = np.random.choice(len(a), size=len(a), replace=True)
        sample_diff = np.mean(a[idx] - b[idx])
        diffs.append(sample_diff)

    p = np.mean(np.abs(diffs) >= np.abs(observed))
    return observed, p


for task in ['ames', 'bbbp', 'cyp3a4', 'dili', 'hia', 'pgp']:

    with open(f'saved_improve_{task}.pkl', 'rb') as f:
        data = pickle.load(f)

    scores = data['scores']
    smiles = data['smiles']
    names = data['names']
    new_smiles = data['new_smiles']
    all_blocks = data['all_blocks']

    #print(len(scores[task][0]))

    pre_scores, after_scores = scores[task][0]

    #print(len(np.unique(names)))
    #print(len(np.unique(pre_scores)))
    #for n, s in zip(smiles, pre_scores):
    #    print(n, s)
    #zz

    all_done = set()
    new_nums = []
    for ps, smi in zip(pre_scores, smiles):
        if smi in all_done: 
            #print(smi)
            #zz
            continue
        all_done.add(smi)
        new_nums.append(ps)
    pre_scores = np.array(new_nums)

    print(len(pre_scores), len(after_scores))

    stat, p = ttest_rel(pre_scores, after_scores)
    #t_stat, p_value = ttest_ind(pre_scores, after_scores, equal_var=False)
    stat, p = wilcoxon(pre_scores, after_scores, zero_method='pratt')#, alternative='two-sided')

    print(task)
    print(pre_scores.mean(), after_scores.mean())
    print(f"t-statistic: {stat:.3f}, p-value: {p:.3e}")

    if p < 0.05:
        print("Reject the null hypothesis: there is a significant difference between the paired means.")
    else:
        print("Fail to reject the null: no significant difference between the paired means.")
    print()



