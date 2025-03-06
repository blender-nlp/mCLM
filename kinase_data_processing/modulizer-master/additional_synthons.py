from rdkit import Chem
from rdkit.Chem import AllChem


#methylamid_rxn = AllChem.ReactionFromSmarts('[#6:1][CX3:2](=[OX1:3])[NX3H1:4][CX4H3:5].[OH2:6]>>[#6:1][CX3:2](=[OX1:3])[OH:6].[NX3H2:4][CX4H3:5]')
#methylamid_patt = Chem.MolFromSmarts('[#6][CX3](=[OX1])[NX3H1][CX4H3]')
#water_mol = Chem.MolFromSmiles('O')
tida = Chem.MolFromSmiles('B2OC(=O)C([CH3])([CH3])N([CH3])C([CH3])([CH3])C(=O)O2')
tida_anchor = Chem.MolFromSmarts('cB2OC(=O)C([CH3])([CH3])N([CH3])C([CH3])([CH3])C(=O)O2')
between_amoatic_n = Chem.MolFromSmarts('nc(Br)n')
cbz_tida_anchor = Chem.MolFromSmarts('CN1C(C)(C)C(=O)OB(OC(=O)C1(C)C)c1ccc(COC=O)cc1')
bromide = Chem.MolFromSmiles('Br')
ncb_mol = Chem.MolFromSmarts('[NX3]C(=O)OCc1ccc(cc1)B([OH0])[OH0]')


def multiply_synthons(all_smiles, rxlist, origins, cmd_status, args):
    added = set()
    for smi in all_smiles:
        if smi not in origins:
            print(f"SMILES WITHOUT ORIGINS {smi}")
            continue
        mol = Chem.MolFromSmiles(smi)
        donerx = []
        for rxinfo in rxlist:
            itr = 0
            while mol.HasSubstructMatch(rxinfo['core']):
                prods = rxinfo['rxn'].RunReactants([mol, *rxinfo['fixedmols']])
                prod = prods[0][0]
                Chem.SanitizeMol(prod)
                if prod:
                    mol = prod
                    itr += 1
                else:
                    break
            if itr:
                newsmi = Chem.MolToSmiles(mol)
                if args.debug:
                    print(f"{rxinfo['rxid']} :: {itr} :: {newsmi} from {smi} due to C(=O)NMe hydrolysis")
                if donerx:
                    print("PROBLEM WITH", smi, donerx, rxinfo)
                    print("RXLIST", rxlist)
                    raise NotImplementedError
                donerx.append(rxinfo['rxid'])
                added.add(newsmi)
                if newsmi not in origins:
                    origins[newsmi] = set()
                origins[newsmi].update(origins[smi])
                if newsmi not in cmd_status:
                    cmd_status[newsmi] = set()
                cmd_status[newsmi].update(cmd_status[smi])
    return added, origins, cmd_status


def swap(smi_list, origins, cmd_status):
    res = []
    for smi in smi_list:
        mol_mol = Chem.MolFromSmiles(smi)
        if mol_mol.HasSubstructMatch(ncb_mol):
            continue
        # swap: B, Br ==> B, Br
        hasTida = mol_mol.HasSubstructMatch(tida)
        hasAnchor = mol_mol.HasSubstructMatch(tida_anchor)
        if hasTida and mol_mol.HasSubstructMatch(bromide) and not mol_mol.HasSubstructMatch(between_amoatic_n):
            tmp = Chem.MolFromSmiles('[As]')
            tmp_prod = AllChem.ReplaceSubstructs(mol_mol, tida, tmp)
            tmp_prod2 = AllChem.ReplaceSubstructs(tmp_prod[0], bromide, tida)
            prod = AllChem.ReplaceSubstructs(tmp_prod2[0], tmp, bromide)
            # print("==>", Chem.MolToSmiles(prod[0]), "FROM:", mol)
            if prod[0] not in res:
                newsmi = Chem.MolToSmiles(prod[0])
                res.append(newsmi)
                if newsmi not in origins:
                    origins[newsmi] = set()
                origins[newsmi].update(origins[smi])
                if newsmi not in cmd_status:
                    cmd_status[newsmi] = set()
                cmd_status[newsmi].update(cmd_status[smi])
    return res, origins, cmd_status


def swap_mono(smi_list, origins, cmd_status):
    res = []
    for smi in smi_list:
        if smi not in origins:
            print("SMI NOT IN ORIGINS,", smi, "\nORIGINS:", origins.keys())
            continue
        mol_mol = Chem.MolFromSmiles(smi)
        # Br -> Btida
        if mol_mol.HasSubstructMatch(bromide) and not mol_mol.HasSubstructMatch(between_amoatic_n):
            prod = AllChem.ReplaceSubstructs(mol_mol, bromide, tida)
            prod_smiles = Chem.MolToSmiles(prod[0])
            if prod_smiles not in smi_list and prod_smiles not in res:
                res.append(prod_smiles)
                if prod_smiles not in origins:
                    origins[prod_smiles] = set()
                origins[prod_smiles].update(origins[smi])
                if prod_smiles not in cmd_status:
                    cmd_status[prod_smiles] = set()
                cmd_status[prod_smiles].update(cmd_status[smi])
        # Btida --> Br
        if mol_mol.HasSubstructMatch(tida_anchor) and not mol_mol.HasSubstructMatch(cbz_tida_anchor):
            prod = AllChem.ReplaceSubstructs(mol_mol, tida, bromide)
            prod_smiles = Chem.MolToSmiles(prod[0])
            if prod_smiles not in smi_list and prod_smiles not in res:
                res.append(prod_smiles)
                if prod_smiles not in origins:
                    origins[prod_smiles] = set()
                origins[prod_smiles].update(origins[smi])
                if prod_smiles not in cmd_status:
                    cmd_status[prod_smiles] = set()
                cmd_status[prod_smiles].update(cmd_status[smi])
    return res, origins, cmd_status


def swap_deprotected(mol_list):
    tida = Chem.MolFromSmiles('B(O)O')
    tida_anchor = Chem.MolFromSmarts('cB(O)O')
    between_amoatic_n = Chem.MolFromSmarts('nc(Br)n')
    # cbz_tida_anchor = Chem.MolFromSmarts('CN1C(C)(C)C(=O)OB(0)0')
    bromide = Chem.MolFromSmiles('Br')

    res = []
    for mol in mol_list:
        mol_mol = Chem.MolFromSmiles(mol)
        if mol_mol.HasSubstructMatch(tida_anchor) and mol_mol.HasSubstructMatch(bromide) and not mol_mol.HasSubstructMatch(between_amoatic_n) and not mol_mol.HasSubstructMatch(tida_anchor):
            tmp = Chem.MolFromSmiles('[As]')
            tmp_prod = AllChem.ReplaceSubstructs(mol_mol, tida, tmp)
            tmp_prod2 = AllChem.ReplaceSubstructs(tmp_prod[0], bromide, tida)
            prod = AllChem.ReplaceSubstructs(tmp_prod2[0], tmp, bromide)
            print(Chem.MolToSmiles(prod[0]))
            if prod[0] not in res:
                res.append(Chem.MolToSmiles(prod[0]))
    return res


def swap_mono_deprotected(mol_list):
    tida = Chem.MolFromSmiles('B(O)O')
    tida_anchor = Chem.MolFromSmarts('cB(O)O')
    between_amoatic_n = Chem.MolFromSmarts('nc(Br)n')
    cbz_tida_anchor = Chem.MolFromSmarts('CN1C(C)(C)C(=O)OB(0)0')
    bromide = Chem.MolFromSmiles('Br')

    res = []
    for mol in mol_list:
        mol_mol = Chem.MolFromSmiles(mol)
        if mol_mol.HasSubstructMatch(bromide) and not mol_mol.HasSubstructMatch(between_amoatic_n):
            prod = AllChem.ReplaceSubstructs(mol_mol, bromide, tida)
            prod_smiles = Chem.MolToSmiles(prod[0])
            if prod_smiles not in mol_list and prod_smiles not in res:
                res.append(prod_smiles)
        if mol_mol.HasSubstructMatch(tida_anchor) and not mol_mol.HasSubstructMatch(cbz_tida_anchor):
            prod = AllChem.ReplaceSubstructs(mol_mol, tida, bromide)
            prod_smiles = Chem.MolToSmiles(prod[0])
            if prod_smiles not in mol_list and prod_smiles not in res:
                res.append(prod_smiles)
    return res


def deprotect(mol_list):
    tmp = set()
    rxn = '[CX4:1][NX3:2]([CX4,c:41])[C:34](=[O:39])[O:35][CH2:63][c:36]2[cH:37][cH:38][c:62]([B:88]1[O:4][C:9](=[O:10])[C:5]([CH3:30])([CH3:31])[N:6]([CH3:7])[C:8]([CH3:32])([CH3:33])[C:11](=[O:12])[O:3]1)[cH:60][cH:61]2.[O:40]>>[*:1][NX3:2][*:41].[O:35][C:34](=[O:39])[O:40][CH2:63][c:36]2[c:37][c:38][c:60]([B:88]1[O:4][C:9](=[O:10])[C:5]([CH3:30])([CH3:31])[N:6]([CH3:7])[C:8]([CH3:32])([CH3:33])[C:11](=[O:12])[O:3]1)[c:61][c:62]2'
    # rxn = '[CX4,c:1][NX3H:2][C:34](=[O:39])[O:35][C:36]([CH3:37])([CH3:38])[B:88]1[O:4][C:9](=[O:10])[C:5]([CH3:30])([CH3:31])[N:6]([CH3:7])[C:8]([CH3:32])([CH3:33])[C:11](=[O:12])[O:3]1.[O:40]>>[*:1][NX3:2].[C:34](=[O:39])=[O:35].[O:40][C:36]([CH3:37])([CH3:38])[B:88]1[O:4][C:9](=[O:10])[C:5]([CH3:30])([CH3:31])[N:6]([CH3:7])[C:8]([CH3:32])([CH3:33])[C:11](=[O:12])[O:3]1'
    deprotection = AllChem.ReactionFromSmarts(rxn)
    left_side = rxn.split('>>')
    left_motifs = left_side[0].split('.')
    left0_mol = Chem.MolFromSmarts(left_motifs[0])
    cnc_mol = Chem.MolFromSmarts('[CX4][NX3H][CX4]')
    water_mol = Chem.MolFromSmiles('O')
    for mol in mol_list:
        print(mol)
        if Chem.MolFromSmiles(mol).HasSubstructMatch(left0_mol) and not Chem.MolFromSmiles(mol).HasSubstructMatch(cnc_mol):
            # print('performing reaxction')
            depr_prod = deprotection.RunReactants((Chem.MolFromSmiles(mol), water_mol))
            for pair in depr_prod:
                if pair[0] != '':
                    tmp.add(Chem.MolToSmiles(pair[0]))
    return tmp


def deprotect2(mol_list):
    tmp = set()
    # rxn = '[CX4:1][NX3:2]([CX4,c:41])[C:34](=[O:39])[O:35][C:36]([CH3:37])([CH3:38])[B:88]1[O:4][C:9](=[O:10])[C:5]([CH3:30])([CH3:31])[N:6]([CH3:7])[C:8]([CH3:32])([CH3:33])[C:11](=[O:12])[O:3]1.[O:40]>>[*:1][NX3:2][*:41].[C:34](=[O:39])=[O:35].[O:40][C:36]([CH3:37])([CH3:38])[B:88]1[O:4][C:9](=[O:10])[C:5]([CH3:30])([CH3:31])[N:6]([CH3:7])[C:8]([CH3:32])([CH3:33])[C:11](=[O:12])[O:3]1'
    rxn = '[CX4,c:1][NX3H:2][C:34](=[O:39])[O:35][C:63][c:36]2[cH:37][cH:38][c:62]([B:88]1[O:4][C:9](=[O:10])[C:5]([CH3:30])([CH3:31])[N:6]([CH3:7])[C:8]([CH3:32])([CH3:33])[C:11](=[O:12])[O:3]1)[cH:60][cH:61]2.[O:40]>>[*:1][NX3:2].[O:35][C:34](=[O:39])[O:40][C:63][c:36]2[c:37][c:38][c:60]([B:88]1[O:4][C:9](=[O:10])[C:5]([CH3:30])([CH3:31])[N:6]([CH3:7])[C:8]([CH3:32])([CH3:33])[C:11](=[O:12])[O:3]1)[c:61][c:62]2'

    deprotection = AllChem.ReactionFromSmarts(rxn)
    left_side = rxn.split('>>')
    left_motifs = left_side[0].split('.')
    left0_mol = Chem.MolFromSmarts(left_motifs[0])
    cn_mol = Chem.MolFromSmarts('[CX4,c,OX2][NX3H2]')
    water_mol = Chem.MolFromSmiles('O')
    for mol in mol_list:
        # print(mol)
        if Chem.MolFromSmiles(mol).HasSubstructMatch(left0_mol) and not Chem.MolFromSmiles(mol).HasSubstructMatch(cn_mol):
            # print('performing reaxction')
            depr_prod = deprotection.RunReactants((Chem.MolFromSmiles(mol), water_mol))
            for pair in depr_prod:
                if pair[0] != '':
                    # print('DEPROTECTED PRIMARY AMINE',Chem.MolToSmiles(pair[0]))
                    tmp.add(Chem.MolToSmiles(pair[0]))
    return tmp


def deprotect_bmida(mol_list):
    tmp = set()
    # rxn = '[CX4:1][NX3:2]([CX4,c:41])[C:34](=[O:39])[O:35][C:36]([CH3:37])([CH3:38])[B:88]1[O:4][C:9](=[O:10])[C:5]([CH3:30])([CH3:31])[N:6]([CH3:7])[C:8]([CH3:32])([CH3:33])[C:11](=[O:12])[O:3]1.[O:40]>>[*:1][NX3:2][*:41].[C:34](=[O:39])=[O:35].[O:40][C:36]([CH3:37])([CH3:38])[B:88]1[O:4][C:9](=[O:10])[C:5]([CH3:30])([CH3:31])[N:6]([CH3:7])[C:8]([CH3:32])([CH3:33])[C:11](=[O:12])[O:3]1'
    rxn = '[#6:1][B:88]1[O:4][C:9](=[O:10])[C:5]([CH3:30])([CH3:31])[N:6]([CH3:7])[C:8]([CH3:32])([CH3:33])[C:11](=[O:12])[O:3]1.[O:13].[O:14]>>[#6:1][B:88]([OH:13])[OH:14].[O:4][C:9](=[O:10])[C:5]([CH3:30])([CH3:31])[N:6]([CH3:7])[C:8]([CH3:32])([CH3:33])[C:11](=[O:12])[O:3]'

    deprotection = AllChem.ReactionFromSmarts(rxn)
    left_side = rxn.split('>>')
    left_motifs = left_side[0].split('.')
    left0_mol = Chem.MolFromSmarts(left_motifs[0])
    ncb_mol = Chem.MolFromSmarts('[NX3]C(=O)OCc1ccc(cc1)B([OH0])[OH0]')
    for mol in mol_list:
        # print(mol)
        if Chem.MolFromSmiles(mol).HasSubstructMatch(left0_mol) and not Chem.MolFromSmiles(mol).HasSubstructMatch(ncb_mol):
            # print('performing reaxction')
            depr_prod = deprotection.RunReactants((Chem.MolFromSmiles(mol), Chem.MolFromSmiles('O'), Chem.MolFromSmiles('O')))
            for pair in depr_prod:
                if pair[0] != '':
                    # print('DEPROTECTED PRIMARY AMINE',Chem.MolToSmiles(pair[0]))
                    tmp.add(Chem.MolToSmiles(pair[0]))
    return tmp
