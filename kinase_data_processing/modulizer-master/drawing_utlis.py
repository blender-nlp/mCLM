from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole


def smilesListToMol(smiles_list):
    res = []
    for smiles in smiles_list:
        smiles_mod = Chem.MolFromSmiles(smiles)
        res.append(smiles_mod)
    return res


def draw_smiles_list(sm, retros_list):
    # sm = extract_list_from_dict(synthons_sm_only)
    retros_struct = smilesListToMol(retros_list)
    sm_struct = smilesListToMol(sm)

    target_img = Draw.MolsToGridImage(retros_struct, molsPerRow=5, maxMols=150, useSVG=True)
    with open('target_img.svg', 'w') as f:
        f.write(target_img.data)

    img = Draw.MolsToGridImage(sm_struct, molsPerRow=5, maxMols=350, useSVG=True)
    with open('img.svg', 'w') as f:
        f.write(img.data)


def draw_synthons(sm, name):
    sm_struct = smilesListToMol(sm)
    img = Draw.MolsToGridImage(sm_struct, molsPerRow=8, maxMols=500, useSVG=True)
    with open(name, 'w') as f:
        f.write(img.data)


def draw_synthons_mol(sm, name):
    img = Draw.MolsToGridImage(sm, molsPerRow=8, maxMols=500, useSVG=True)
    with open(name, 'w') as f:
        f.write(img.data)


def splitRxns(list):
    IPythonConsole.ipython_useSVG = True
    im_tmp = []
    for rxn in list:
        entry = rxn.split('>>')
        entry_j = '.'.join(entry)
        im_tmp.append(entry_j)
    return im_tmp


def convert_to_smiles_most_popular_sm(list_of_mols):
    # k = 0
    initial = True
    toDraw = []
    list_of_mols_sorted = sorted(list_of_mols, key=lambda tup: tup[0], reverse=True)
    maximal = None
    for entry in list_of_mols_sorted:
        # print(entry)
        if initial:
            maximal = entry[0][0]
            initial = False
        if maximal == entry[0][0] and maximal > 1:
            # print(entry[0])
            toDraw.append(Chem.MolFromSmiles(entry[0][1]))
        # k += 1
    if len(toDraw) == 0:
        print('No coomn synthon for this set')
    return toDraw


def draw_results(swapped_mono, swapped, res1, res2, res3, res4, res5):
    if len(res1) > 0:
        drawing_utlis.draw_synthons(res1, 'Monofunctional.svg')
        if len(swapped_mono) > 0:
            drawing_utlis.draw_synthons(swapped_mono, 'Swapped_mono.svg')
        else:
            print('no swapped mono')
    if len(res2) > 0:
        drawing_utlis.draw_synthons(res2, 'Difunctional_prefilters.svg')
        swapped = additional_synthons.swap(res2)
        if len(swapped) > 0:
            drawing_utlis.draw_synthons(swapped, 'Swapped.svg')
        else:
            print('no swapped')
    if len(res3) > 0:
        drawing_utlis.draw_synthons(res3, 'Invalid.svg')
    if len(res4) > 0:
        drawing_utlis.draw_synthons(res4, 'Extra_group.svg')
    if len(res5) > 0:
        drawing_utlis.draw_synthons(res5, 'More_rings.svg')
