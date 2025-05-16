

import pickle

from collections import defaultdict

from rdkit import Chem

from rdkit.Chem import Draw
from PIL import Image, ImageDraw, ImageFont



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


for task in ['ames', 'bbbp', 'cyp3a4', 'dili', 'hia', 'pgp']:

    with open(f'saved_improve_{task}.pkl', 'rb') as f:
        data = pickle.load(f)

    scores = data['scores']
    smiles = data['smiles']
    names = data['names']
    new_smiles = data['new_smiles']
    all_blocks = data['all_blocks']


    before_freq = defaultdict(int)
    after_freq = defaultdict(int)

    for bef, aft in all_blocks[task]:
        bef = bef[0]

        for be in bef.split('.'):
            for b in be.split('^'):
                before_freq[b] += 1
                
        for af in aft.split('.'):
            for a in af.split('^'):
                after_freq[a] += 1




        # Compute the difference for all keys
        all_keys = set(before_freq) | set(after_freq)
        diffs = {key: after_freq[key] - before_freq[key] for key in all_keys}

        # Sort by difference value
        sorted_diffs = sorted(diffs.items(), key=lambda x: x[1], reverse=True)

    print('Task:', task)

    # Print top 10 increases
    print("Top 10 increases:")
    for key, diff in sorted_diffs[:10]:
        print(f"{key}: {diff}")

    # Print bottom 10 decreases
    print("\nBottom 10 decreases:")
    for key, diff in sorted_diffs[-10:]:
        print(f"{key}: {diff}")

    print()

    
    img = smiles_to_image([m for m, _ in sorted_diffs[:5]], grid_size=(1,5))
    img.save(f'library_images/{task}_top.png')


    img = smiles_to_image([m for m, _ in sorted_diffs[-5:]][::-1], grid_size=(1,5))
    img.save(f'library_images/{task}_bottom.png')


