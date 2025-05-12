
import numpy as np

from rdkit import Chem

from rdkit.Chem.Crippen import MolLogP

from sklearn.metrics import mean_squared_error

from itertools import product

import matplotlib.pyplot as plt

import pygad

import sys


from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
import io

from scipy.stats import wasserstein_distance

from kinase_data_processing.blocks_to_mol import join_fragments

max_val = 10
min_val = -3
points = 14


library_size = 8
molecule_size = 5

values = np.linspace(min_val, max_val, points)

score_func = mean_squared_error


library = open('kinase_data_processing/vocabulary.txt').readlines()
library = [b.strip() for b in library]

di_blocks = set([b for b in library if '1*' in b and '2*' in b])
mono_blocks = set([b.replace('1*', '*').replace('2*', '*') for b in library if ('1*' in b or '2*' in b) and not ('1*' in b and '2*' in b)])

total_blocks = list(mono_blocks.union(di_blocks))


def build_set(blocks):

    mblocks = [b for b in blocks if b in mono_blocks]
    dblocks = [b for b in blocks if b in di_blocks]

    possible_mols = set()

    for start in mblocks:
        for i in range(molecule_size - 2 + 1):
            for dbs in product(dblocks, repeat=i):
                for end in mblocks:
                    s = start.replace('*', '1*')
                    e = end.replace('*', '2*')
                    new_mol = join_fragments('.'.join([s] + list(dbs) + [e]))

                    possible_mols.add(new_mol)
                    
    return possible_mols

#print(list(mono_blocks)[:2] + list(di_blocks)[:2])
#built_library = build_set(list(mono_blocks)[:2] + list(di_blocks)[:2])
#built_smis = [Chem.MolToSmiles(mol) for mol in built_library]


def histo(mols):

    props = [MolLogP(mol) for mol in mols]

    #print(props)

    fig = plt.figure(figsize=(8, 6))
    plt.hist(props, bins=30, color='blue', edgecolor='black', alpha=0.7, density = True)

    # Labels and title
    plt.xlabel('XLogP')
    plt.ylabel('Frequency')
    #plt.title('Histogram of Randomly Generated Data')

    # Show plot
    return fig

#fig = histo(built_library)
#fig.savefig('library_selection/histo.png')

#zz


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def molecule_score(mol):

    val = MolLogP(mol)

    #nearest = find_nearest(values, val)
    #print(val, nearest)
    return val # score_func([val], [nearest])




def build_input(X):
    
    blocks = [total_blocks[i] for i in X]

    built_library = build_set(blocks)

    return built_library


BAD_SCORE = 10


all_scores = [[]]


def f(ga_instance, X, X_idx):
    #print('f called at:', X_idx)
    inp = build_input(X)

    if len(inp) != 0:

        pred_vals = [molecule_score(mol) for mol in inp]

        #Check how close we are to each element in the list
        if True: #use wasserstein
            score = wasserstein_distance(values, pred_vals)
        else:
            scores = []
            for v in values:
                nearest = find_nearest(pred_vals, v)
                scores.append(score_func([v], [nearest]))

            score = np.nanmean(scores)
            if np.isnan(score): score = BAD_SCORE
    else:
        score = BAD_SCORE


    #print('score:', score)
    print(X_idx,":", score)

    all_scores[-1].append(score)

    return - score




fitness_function = f

seed = 42

num_generations = 20
num_parents_mating = 4

sol_per_pop = 16
num_genes = library_size
gene_space = np.arange(len(total_blocks))

parent_selection_type = "sss"
keep_parents = 1

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 10

def callback_gen(ga_instance):
    print("Generation : ", ga_instance.generations_completed)
    sys.stdout.flush()
    all_scores.append([])
    #print("Fitness of the best solution :", ga_instance.best_solution()[1])

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       gene_space=gene_space,
                       allow_duplicate_genes=False,
                       gene_type=int,
                       on_generation=callback_gen,
                       random_seed=seed,
                       save_best_solutions=True,
                       save_solutions=True,
                       )

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))


ga_instance.plot_fitness(plot_type="scatter", title='', ylabel='Fitness (-Wasserstein)', save_dir='library_selection/plots/fitness.png')
ga_instance.plot_new_solution_rate(save_dir='library_selection/plots/new_sol_rate.png')

print('All Scores:')
print(all_scores)

#np.set_printoptions(threshold=sys.maxsize)
#print(ga_instance.best_solutions)




def smiles_to_image(smiles_list, img_size=(200, 200), grid_size=None):
    """
    Generates an image for each SMILES string and combines them into one.
    :param smiles_list: List of SMILES strings
    :param img_size: Tuple (width, height) for each molecule image
    :param grid_size: Tuple (rows, cols) for arranging images; if None, it will be determined automatically
    :return: Combined image
    """
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    valid_mols = [mol for mol in mols if mol is not None]
    
    if not valid_mols:
        raise ValueError("No valid molecules generated from SMILES.")
    
    # Determine grid size automatically if not provided
    n = len(valid_mols)
    if grid_size is None:
        cols = int(n**0.5) + 1
        rows = (n + cols - 1) // cols  # Ensure enough rows
    else:
        rows, cols = grid_size
    
    # Draw individual molecule images
    images = [Draw.MolToImage(mol, size=img_size) for mol in valid_mols]
    
    # Create a blank canvas
    total_width = cols * img_size[0]
    total_height = rows * img_size[1]
    combined_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    
    # Paste images onto the canvas
    for idx, img in enumerate(images):
        row, col = divmod(idx, cols)
        x_offset = col * img_size[0]
        y_offset = row * img_size[1]
        combined_image.paste(img, (x_offset, y_offset))
    
    return combined_image

# Example usage
if False:#__name__ == "__main__":
    smiles_list = ["CCO", "CCN", "CCCC", "c1ccccc1", "C1=CC=CN=C1"]
    img = smiles_to_image(smiles_list, grid_size=(1,library_size))
    #img.show()  # Show the combined image
    img.save("molecules.png")  # Save the image


block_sols = [[total_blocks[i] for i in X] for X in ga_instance.best_solutions]

for i, sol in enumerate(block_sols):
    built_library = build_set(sol)
    fig = histo(built_library)
    norm = 1/(max_val-min_val)
    plt.plot([min_val, min_val],[0,norm], 'k')
    plt.plot([max_val, max_val],[0,norm], 'k')
    plt.plot([min_val, max_val],[norm,norm], 'k')
    fig.savefig(f'library_selection/plots/{i}_histo.png')

    img = smiles_to_image(sol, grid_size=(1,library_size))
    img.save(f'library_selection/plots/{i}_blocks.png')  # Save the image



