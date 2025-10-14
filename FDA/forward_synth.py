
from rdkit import Chem

import multiprocessing
from itertools import permutations
from rdkit.Chem import rdChemReactions

debug = False

def canonicalize(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        #Chem.RemoveStereochemistry(mol)
        #Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except:
        return None


def run_with_timeout(timeout, func, *args, **kwargs):
    def wrapper(queue, *args, **kwargs):
        queue.put(func(*args, **kwargs))

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=wrapper, args=(queue, *args), kwargs=kwargs)
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError("Function execution timed out")

    return queue.get() if not queue.empty() else None


nuc_environ = [
    '[#6]-[B][O]',
    '[#7H2,#7H1;!$([#7]S(=O));!$([#7][#6](=O))]',
    '[#6^2;!$([#6](=[#8]));$([c;r6][c;r6;d2][c;r6;d2][c;r6;d2][c;r6;d2][c;r6;d2])]-[OH1]'
]
el_environ = [
    '[#6](=[#8])[#8H1]',
    '[#6][Br]',
    'N=C=O'
]

reaction_rules = [
    '[#6:1]-[#5][#8].[#6:2][Br]>>[#6:1]-[#6:2]',
    '[#7H2:1].[#6:2][Br]>>[#7H1:1]-[#6:2]',
    '[#7H2:1].[#6:2](=[O:3])[OH1]>>[#7:1]-[#6:2](=[O:3])',
    '[#7H1;!$([#7]S(=O));!$([#7][#6](=O)):1].[#6:2][Br]>>[#7H0:1]-[#6:2]',
    '[#7H1;!$([#7]S(=O));!$([#7][#6](=O)):1].[#6:2](=[O:3])[OH1]>>[#7:1]-[#6:2](=[O:3])',
    '[#7H2:1].[*:2][N:3]=[C:4]=[O:5]>>[#7:1]-[C:4](=[O:5])[N:3][*:2]',
    '[#7H1;!$([#7]S(=O));!$([#7][#6](=O)):1].[*:2][N:3]=[C:4]=[O:5]>>[#7:1]-[C:4](=[O:5])[N:3][*:2]',
    '[#6^2;!$([#6](=[#8])):1]-[OH1:2].[#6:3][Br]>>[#6^2;!$([#6](=[#8])):1]-[O:2]-[#6:3]'
]
deprotection_rules = [
    '[C;d4;$(COC(=O)[#7H1;!$([1#7])]):1][CX4]>>[C;$(COC(=O)[#7H1;!$([1#7])]):1]',
    '[1#7H1:1]C(=O)O[C;d4;X4][CX4]>>[1#7H2:1]',
    '[1#7H0:1]C(=O)O[C;d4;X4][CX4]>>[1#7H1:1]',
    '[#6:1]-[1#6]>>[#6:1]-[Cl]',
    '[#6:1]-[4#6]>>[#6:1]-[Br]',
    '[#6X2:1]-[#14]>>[#6X2:1]',
    '[o;r5;d2:1][c;r5;d3:2][n;r5;d2:3][c;r5;d2][c;r5;d2]>>[C:2](=[O:1])[N:3]'
]

convert_rules = [
    '[#6:1]-[#5][#8]>>[#6:1]-[1*]',
    '[#7H2:1]>>[#7H1:1]-[1*]',
    '[#7H1;!$([#7]S(=O));!$([#7][#6](=O));!$([#7][1*]):1]>>[#7H0:1]-[1*]',
    '[#6^2;!$([#6](=[#8])):1]-[OH1:2]>>[#6^2;!$([#6](=[#8])):1]-[O:2]-[1*]',
    '[#6:1][Br]>>[#6:1]-[2*]',
    '[#6:1](=[O:2])[OH1]>>[#6:1](=[O:2])[2*]',
    '[*:2][N:3]=[C:4]=[O:5]>>[*:2][N:3][C:4](=[O:5])-[2*]',
    '[C;d4;$(COC(=O)[#7H1;!$([1#7])]):1][CX4]>>[C;$(COC(=O)[#7H1;!$([1#7])]):1]',
    '[1#7H1:1]C(=O)O[C;d4;X4][CX4]>>[1#7H2:1]',
    '[1#7H0:1]C(=O)O[C;d4;X4][CX4]>>[1#7H1:1]',
    '[#6:1]-[1#6]>>[#6:1]-[Cl]',
    '[#6:1]-[4#6]>>[#6:1]-[Br]',
    '[#6X2:1]-[#14]>>[#6X2:1]',
    '[o;r5;d2:1][c;r5;d3:2][n;r5;d2:3][c;r5;d2][c;r5;d2]>>[C:2](=[O:1])[N:3]'
]

def classify_building_blocks(building_blocks, nuc_environ, el_environ):
    nucleophiles, bifunctionals, electrophiles = [], [], []
    
    for smi in building_blocks:
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            continue  # Skip invalid molecules
        
        has_nuc = any(mol.HasSubstructMatch(Chem.MolFromSmarts(pat)) for pat in nuc_environ)
        has_el = any(mol.HasSubstructMatch(Chem.MolFromSmarts(pat)) for pat in el_environ)
        
        if has_nuc and has_el:
            bifunctionals.append(smi)
        elif has_nuc:
            nucleophiles.append(smi)
        elif has_el:
            electrophiles.append(smi)
    
    if debug: print("Building block classification:")
    if debug: print(f"  Nucleophiles: {nucleophiles}")
    if debug: print(f"  Bifunctionals: {bifunctionals}")
    if debug: print(f"  Electrophiles: {electrophiles}")
    
    return nucleophiles, bifunctionals, electrophiles

def apply_reaction(reactants, reaction_smarts):
    try:
        rxn = rdChemReactions.ReactionFromSmarts(reaction_smarts)
        expected_reactants = rxn.GetNumReactantTemplates()
        if len(reactants) != expected_reactants:
            return None

        mols = [Chem.MolFromSmiles(smi) for smi in reactants]
        if None in mols:
            return None

        products = rxn.RunReactants(mols)
        result = [Chem.MolToSmiles(p[0]) for p in products if p]
        return result if result else None
    except Exception as e:
        return None

def apply_f_deprotection(molecules, deprotection_rules):
    new_molecules = molecules
    for rule in deprotection_rules:
        while True:
            old_molecules = new_molecules
            products = apply_reaction(new_molecules, rule)
            if products:
                new_molecules = [products[0]]
            if new_molecules == old_molecules or products == None:
                break
    deprotected_molecules = new_molecules
    return deprotected_molecules


def without_isotopes(mol):
    if mol is None:
        return Chem.MolFromSmiles('')
    atom_data = [(atom, atom.GetIsotope()) for atom in mol.GetAtoms()]
    for atom, isotope in atom_data:
      if atom.GetAtomicNum() == 7 and isotope == 1:
          atom.SetIsotope(0)
    out = mol
    return out

def convert_format(result, convert_rules):
    converted = []
    for path in result:
        mols = []
        for mol in path:
            deprot = without_isotopes(Chem.MolFromSmiles(apply_f_deprotection([mol], convert_rules)[0]))
            mols.append(Chem.MolToSmiles(deprot))
        mols = ('^').join(mols)
        converted.append(mols)
    return converted


def forward_synthesis(target_smiles, building_blocks, reaction_rules, deprotection_rules, convert_rules, max_steps=5, convert=True):
    nucleophiles, bifunctionals, electrophiles = classify_building_blocks(building_blocks, nuc_environ, el_environ)
    
    if not nucleophiles or not electrophiles:
        if debug: print("Insufficient nucleophiles or electrophiles for synthesis.")
        return []  # Need at least one nucleophile and electrophile

    if Chem.MolFromSmiles(target_smiles).HasSubstructMatch(Chem.MolFromSmarts('I')):
        deprotection_rules[2] = '[#6:1]-[1#6]>>[#6:1]-[I]'
        convert_rules[4] = '[#6:1]-[1#6]>>[#6:1]-[I]'

    if not(Chem.MolFromSmiles(target_smiles).HasSubstructMatch(Chem.MolFromSmarts('[#7H1:1]C(=O)O[C;d4;X4][CX4]')) or Chem.MolFromSmiles(target_smiles).HasSubstructMatch(Chem.MolFromSmarts('[#7H0:1]C(=O)O[C;d4;X4][CX4]'))):
        deprotection_rules[1] = '[#7H1:1]C(=O)O[C;d4;X4][CX4]>>[#7H2:1]'
        deprotection_rules[2] = '[#7H0:1]C(=O)O[C;d4;X4][CX4]>>[#7H1:1]'
        convert_rules[8] = '[#7H1:1]C(=O)O[C;d4;X4][CX4]>>[#7H2:1]'
        convert_rules[9] = '[#7H0:1]C(=O)O[C;d4;X4][CX4]>>[#7H1:1]'
        
    valid_combinations = []
    
    for step in range(2, min(max_steps+1, len(building_blocks)+1)):
        if step == 2 and (not nucleophiles or not electrophiles):
            break  # Need both for a 2-component reaction
        if step > 2 and len(bifunctionals) < step - 2:
            break  # Need enough bifunctionals
            
        if debug: print(f"Checking {step}-component reactions...")
        for n in nucleophiles:
            for e in electrophiles:
                for bifunc_combo in permutations(bifunctionals, step - 2):
                    reactant_order = [n] + list(bifunc_combo) + [e]
                    current_mol = reactant_order[0]
                    for next_mol in reactant_order[1:]:
                        for rxn_smart in reaction_rules:
                            products = apply_reaction([current_mol, next_mol], rxn_smart)
                            if products:
                                current_mol = products[0]
                                break

                    
                    deprotected_products = apply_f_deprotection([current_mol], deprotection_rules)
                    if Chem.MolFromSmiles(target_smiles).HasSubstructMatch(Chem.MolFromSmarts(deprotected_products[0])) and Chem.MolFromSmiles(deprotected_products[0]).HasSubstructMatch(Chem.MolFromSmarts(target_smiles)):
                        valid_combinations.append(reactant_order)
                        if debug: print(f"Successful synthesis path found: {reactant_order} → {target_smiles}")
        
        if valid_combinations:
            break # Stop searching once a valid path is found // could be removed for a more thorough search

    if valid_combinations == []:
        if debug: print("No valid synthesis found.")
    else:
        if debug: print("\nValid synthesis paths:")
        for path in valid_combinations:
            if debug: print(" + ".join(path), "→", target_smiles)

    if convert:
        return convert_format(valid_combinations, convert_rules)

    return valid_combinations
