#!/usr/bin/env python
# coding: utf-8
import sys, time
import traceback
from functools import partial
from multiprocessing import Process, Queue
import statistics
from rdkit import Chem
from rdkit.Chem import AllChem, rdqueries
# import drawing_utlis
import rxn_mols_prep
import additional_synthons
import inc_and_sel
import loader
smiles_killer = set()
if smiles_killer:
    print(f"SMILESKILLER: {'.'.join(smiles_killer)}", file=sys.stderr)
black_list = {'CN1C(C)(C)C(=O)OB(C=C2C(=O)Nc3ccc(F)cc32)OC(=O)C1(C)C', 'CN1C(C)(C)C(=O)OB(C=Cc2ccccn2)OC(=O)C1(C)C',
              'CN1C(C)(C)C(=O)OB(C=CBr)OC(=O)C1(C)C',
              'CC/C=C/CC(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](C)C(=O)N[C@H](C[16C](=O)OC(C)(C)C)C(=O)N[C@@H](C)C(=O)N[C@H](C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@H](C(=O)N[C@@H](CC(N)=O)C(=O)N[C@@H](CO)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](CCCC[1NH2])C(=O)N[C@H](C(=O)N[C@@H](CC(C)C)C(=O)NCC(=O)N[C@@H](CCC(N)=O)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CO)C(=O)N[C@@H](C)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](CCCC[1NH2])C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CCC(N)=O)C(=O)N[C@H](C[16C](=O)OC(C)(C)C)C(=O)N[C@H](C(=O)N[C@@H](CCSC)C(=O)N[C@@H](CO)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](CCC(N)=O)C(=O)N[C@@H](CCC(N)=O)C(=O)NCC(=O)N[C@@H](CCC(N)=O)C(=O)N[C@@H](CO)C(=O)N[C@@H](CC(N)=O)C(=O)N[C@@H](CCC(N)=O)C(=O)N[C@@H](CCCNC(=N)N)C(=O)NCC(=O)N[C@@H](C)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](C)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](CC(C)C)C(N)=O)C(C)CC)C(C)C)C(C)O)C(C)CC',
              'CNC(=O)c1ccccc1Sc1ccc2c(C=CB3OC(=O)C(C)(C)N(C)C(C)(C)C(=O)O3)nn(C(=O)OC(C)(C)C)c2c1',
              'CN1C(C)(C)C(=O)OB(C=C2C(=O)Nc3ccc(F)cc32)OC(=O)C1(C)C'}

wrong_stereo_smiles = {r'CN1C(C)(C)C(=O)OB(/C=C\c2ccccn2)OC(=O)C1(C)C', r'Br/C=C\c1ccccn1',
                       r'CN1C(C)(C)C(=O)OB(/C=C\c2ccccn2)OC(=O)C1(C)C', r'CN1C(C)(C)C(=O)OB(/C=C\Br)OC(=O)C1(C)C',
                       r'CNC(=O)c1ccccc1Sc1ccc2c(/C=C\Br)nn(C(=O)OC(C)(C)C)c2c1',
                       'CN1C(C)(C)C(=O)OB(C=C2C(=O)Nc3ccc(F)cc32)OC(=O)C1(C)C',
                       r'CNC(=O)c1ccccc1Sc1ccc2c(/C=C\B3OC(=O)C(C)(C)N(C)C(C)(C)C(=O)O3)nn(C(=O)OC(C)(C)C)c2c1'}

# dont add to fg_smarts - in some place in code we do for k in fg_mols
fg_smarts = {'bmida': '[#6][B]2OC(=O)C([CH3])([CH3])[N]([CH3])C([CH3])([CH3])C(=O)O2',
             'COOH': '[#6]C(=O)[OH]',
             'aromBr': 'cBr',
             'CBr': '[CX4H1,CX4H2]Br',
             'hydroxyamine': '[CX4,c][OX2][NH2]',
             'secN': 'CN1C(C)(C)C(=O)OB(OC(=O)C1(C)C)c1ccc(COC([NX3]([CX4H2,CH3])[CX4H2,CH3,$([CX4H]([NX3])([#6])[#6])])=O)cc1',
             'primN': 'CN1C(C)(C)C(=O)OB(OC(=O)C1(C)C)c1ccc(COC([NX3H][CX4,c,$([CX3]1=[NX2]cc[CX3](=[NX2])[NX3]1)])=O)cc1',
             'heteroaroN': '[nH]',
             'vinylBr': '[CX3]=[CX3]Br',
             'O=C=N': '[CX4,c,CX3,SX4][NX2]=[CX2]=[OX1]',
             'phenols': '[c][OX2H1]',
             'aromHalo': 'c[I,Cl]',
             'vinylHalo': '[CX3]=[CX3][I,Cl]',
             'diazoBoronGeneral': 'nc([B,B-])n',
             'diazoBoronR6': 'n[cr6]([B,B-])n',
             }
fg_mols = {k: Chem.MolFromSmarts(fg_smarts[k]) for k in fg_smarts}
cbzmol = Chem.MolFromSmarts('[NX3][C](=[O])O[CH2]c1ccc([BX3])cc1')
wrong_cOH_smarts = ['[cr5][OH]', '[a]1[a][n][a][a][c]1[OH]', '[a]1[a][a][a][n][c]1[OH]', '[a]1[a][a][n][a][c]1[OH]']
wrong_cOH_mols = [Chem.MolFromSmarts(sma) for sma in wrong_cOH_smarts]
# for stereo checking
stereo_to_check_smarts = ('[CX3]=[CX3][BX3]([OX2])([OX2])', '[CX3]=[CX3][Br]')
stereo_to_check_mols = [Chem.MolFromSmarts(sma) for sma in stereo_to_check_smarts]
alkylBr_patt = Chem.MolFromSmarts('[CX4H1,CX4H2]Br')
alkylB_patt = Chem.MolFromSmarts('[CX4H1,CX4H2][BX3,BX4-]')


def _has_forbidden_phenols(smi):
    mol = Chem.MolFromSmiles(smi)
    for wrong_cOH in wrong_cOH_mols:
        if mol.HasSubstructMatch(wrong_cOH):
            return True
    return False


def identifyValidRxns(rxn_list, retros_list, debug=False):
    """
    args:
         rxn_list - list of reaction information, each element of list is list/tuple
         retors_list - list of smiles
    return:
        [rxinfo, ..] where rxinfo = [rxid, rxsmarts, substrate_smiles, inco_list]
    """
    retrons_of_generation = []
    used = set()
    for rxn_line in rxn_list:  # ['retro']:
        # 0:id; 1:rxsma; 2:.Br; 3:?; 4:[c][NH2]; 5:forwardrx
        # info = {'rxid': line[0], 'rxname': line[1], 'rxsmarts': line[2], 'fix': line[3],
        #            'inco': line[4], 'nonselect': line[5], 'forwardsmarts': line[6],
        #            'retrorx': AllChem.ReactionFromSmarts(line[2])}
        small_retrons = rxn_line['fix'].split('.')
        if not small_retrons[-1] and len(small_retrons) == 2:  # no fix
            small_retrons = small_retrons[:-1]
        small_retrons_oryg = small_retrons.copy()
        rxn_splitted = rxn_line['rxsmarts'].split('>>')
        retros_splitted = rxn_splitted[0].split('.')
        retron_smarts = Chem.MolFromSmarts(retros_splitted[0])
        inc_list = rxn_line['inco'].split('.')
        non_sel_list = rxn_line['nonselect'].split('.')
        left_side = rxn_line['forwardsmarts']

        for retron in retros_list:
            # if retron == 'CN1C(C)(C)C(=O)OB(c2ccc(COC(=O)Nc3ncnc4ccc(-c5ccc(CN(CCS(C)(=O)=O)C(=O)OC(C)(C)C)o5)cc34)cc2)OC(=O)C1(C)C':
            #    debug = True
            # else:
            #    debug = False
            retron_mol = Chem.MolFromSmiles(retron)
            if not retron_mol:
                print("CANNOT PARSE RETRON", retron)
                continue
            if not retron_mol.HasSubstructMatch(retron_smarts):
                if debug:
                    print(f"no match to rx: {retron} to {retros_splitted[0]} in rx: {rxn_splitted}")
                continue
            inc_detected = inc_and_sel.identify_incomatibilities(retron_mol, inc_list, retron_smarts)
            if inc_detected:
                if debug:
                    print(f"inco detected for: {retron}")
                continue
            if identify_group(retron_mol, non_sel_list):  # potentially nonselective
                # second_retron = rxn_line['fix'].split('.')
                is_rxn_selective = inc_and_sel.is_selective(rxn_line['rxsmarts'], non_sel_list, left_side, retron_mol, small_retrons_oryg)
                if is_rxn_selective:
                    small_retrons[0] = retron  # 1st elem is empty in db; str starts with dot
                    tmp = []
                    for cmpd in small_retrons:
                        cmpd_mol = Chem.MolFromSmiles(cmpd)
                        tmp.append(cmpd_mol)
                    retron_tuple = tuple(tmp)
                    if debug:
                        print(f"match__sel_check: {rxn_line}")
                    retrons_of_generation.append([rxn_line['rxid'], rxn_line['retrorx'], retron_tuple, inc_list, rxn_line])
                    used.add(retron)
                else:
                    if debug:
                        print(f"no selective detected for: {retron}")

            else:
                small_retrons[0] = retron  # 1st elem was empty
                tmp = []
                for cmpd in small_retrons:
                    cmpd_mol = Chem.MolFromSmiles(cmpd)
                    tmp.append(cmpd_mol)
                retron_tuple = tuple(tmp)
                if debug:
                    print(f"match_no_sel_check: {rxn_line}")
                retrons_of_generation.append([rxn_line['rxid'], rxn_line['retrorx'], retron_tuple, inc_list, rxn_line])
                used.add(retron)
    return retrons_of_generation, used


def _is_stereo_ok(smiles, mols, ret, debug):
    br_smiles = [s for s in smiles if 'Br' in s]
    b_smiles = [s for s in smiles if s.count('B') > s.count('Br')]
    if not(br_smiles and b_smiles):
        return True
    br_db_smiles = [s for s in br_smiles if s.count('=') > s.count('=O')]
    b_db_smiles = [s for s in b_smiles if s.count('=') > s.count('=O')]
    if not(br_db_smiles or b_db_smiles):
        return True
    br_mols = [m for m in mols if m.HasSubstructMatch(stereo_to_check_mols[1])]
    b_mols = [m for m in mols if m.HasSubstructMatch(stereo_to_check_mols[0])]
    if not(br_mols or b_mols):
        return True
    if debug:
        print("POTENTIAL_PROBLEM", '.'.join(smiles), "FROM", Chem.MolToSmiles(ret[2][0]))
    if any([s in wrong_stereo_smiles for s in smiles]):
        if debug:
            print("REMOVED", smiles)
        return False
    return True


def get_num_atoms(mol):
    #natoms = mol.GetNumAtoms()
    natoms = mol.GetNumHeavyAtoms()
    # print("NUM", natoms)
    # then we can add here checking if there is H or protecting group or whatever
    return natoms


def is_all_blocks_valid(mol_list, threshold=5):
    # alkylBr_patt = Chem.MolFromSmarts('[CX4H1,CX4H2]Br')
    # alkylB_patt = Chem.MolFromSmarts('[CX4H1,CX4H2][BX3,BX4-]')
    mol_to_check = [m for m in mol_list if m.HasSubstructMatch(alkylBr_patt) or m.HasSubstructMatch(alkylB_patt)]
    if not mol_to_check:
        return True
    mol_wrong = [m for m in mol_to_check if count_carbons_from_mol(m, exclude_protection=True) <= threshold]
    if mol_wrong:
        # print("WRONG", [Chem.MolToSmiles(m) for m in mol_wrong], "||", [Chem.MolToSmiles(m) for m in mol_list])
        return False
    return True


def performFirstStep(valid_rxns, protection_reactions, timelimit=0, debug=False):
    # update: mol_dict, rxns, synthons
    results = dict()
    empty = set()
    t0 = 0
    if timelimit:
        t0 = time.time()
    for ret in valid_rxns:
        rxn = ret[1]
        rxinfo = ret[-1]
        try:
            ps = rxn.RunReactants(ret[2])
        except:
            print("CANNOT EXECUTE RX", AllChem.ReactionToSmarts(rxn), "ON:", [Chem.MolToSmiles(m) for m in ret[2]])
            raise
        pairs_set = set()
        retron = Chem.MolToSmiles(ret[2][0])
        cmds = dict()
        for p in ps:
            if t0:
                if (time.time() - t0) > timelimit:
                    print("timeout reached inside")
                    raise TimeoutError
            pairs = []
            pair_mols = []
            if rxinfo.get('saveprod', []):
                allowedPoz = rxinfo['saveprod']
                p = [mol for poz, mol in enumerate(p) if poz in allowedPoz]
            if rxinfo.get('minsize', 0):
                minsize = rxinfo['minsize']
                initial_allowed = len(p)
                p = [mol for mol in p if get_num_atoms(mol) > minsize]
                if len(p) != initial_allowed:
                    continue
            for cmpd in p:
                try:
                    Chem.SanitizeMol(cmpd)
                except:
                    print("CANNOT SANITIZE MOL", Chem.MolToSmiles(cmpd, False))
                    print("FORMED IN RX", AllChem.ReactionToSmarts(rxn), "FOR", [Chem.MolToSmiles(r) for r in ret[2]])
                    raise
            if not is_all_blocks_valid(p):
                if debug:
                    print("==ignore::", [Chem.MolToSmiles(s) for s in p])
                continue
            else:
                if debug:
                    print("ALLOWED", [Chem.MolToSmiles(s) for s in p], "FFF", ps)
            for cmpd in p:
                smiles = Chem.MolToSmiles(cmpd)
                for smi in smiles.split('.'):
                    mol = Chem.MolFromSmiles(smi)
                    mol, _ = perform_protection_singlemol(mol, protection_reactions, debug)
                    smi = Chem.MolToSmiles(mol)
                    pairs.append(smi)
                    pair_mols.append(mol)
            if not _is_stereo_ok(pairs, pair_mols, ret, debug):
                continue
            pairs_set.add(tuple(pairs))
            for cmd in pairs:
                if cmd not in cmds:
                    cmds[cmd] = set()
                cmds[cmd].add(tuple(pairs))
        if debug:
            print("RES_of_rdkit", cmds, pairs_set, "FROM", retron, "IN", hash(str(valid_rxns)), ret)
        if not pairs_set and (retron not in results):
            empty.add(retron)
        for cmd in cmds:
            pairs = cmds[cmd]
            if cmd not in results:
                results[cmd] = {'retrons': set(), 'rxns': set(), 'pairs': set(), 'rxids': set()}
            results[cmd]['rxids'].add(ret[0])
            results[cmd]['retrons'].add(retron)
            for pa in pairs:
                rxn = '>>'.join([retron, '.'.join(pa)])
                results[cmd]['pairs'].add(pa)
                results[cmd]['rxns'].add(rxn)
    return results, empty


def identify_group(mol, functional_groups):
    for s in functional_groups:
        if mol.HasSubstructMatch(Chem.MolFromSmarts(s)):
            return True
    return False


def perform_Retro(rxn_retro, rxn_protection, retros_list, debug, max_steps, get_intermediates=False, timeout=0):
    cmd_to_cut = retros_list
    final_synthons = dict()
    initime = time.time()
    nSteps = 0
    intermediates = set()
    retronToDirectSbses = dict()
    allDoneRxes = dict()
    for i in range(max_steps):
        # print(f"CMDTOCUT {i}, {len(cmd_to_cut)} {'.'.join(cmd_to_cut)}")
        valid_rxns_retron, used_retrons = identifyValidRxns(rxn_retro, cmd_to_cut, debug)
        not_used = [smi for smi in cmd_to_cut if smi not in used_retrons]
        timelimit = 0
        if timeout:
            timelimit = timeout - (time.time() - initime)
        step_results, not_reacting_cmds = performFirstStep(valid_rxns_retron, rxn_protection, timelimit, debug)
        if timeout and time.time() - initime > timeout:
            print(f'timeoutREACHED {retros_list} itr: {i} time: {time.time() - initime}')
            raise TimeoutError
        if debug:
            print("RESN_STEP", i, step_results.keys(), "F::", step_results)
            print("RESN_NOREA", i, not_reacting_cmds)
            print("RESN_VALID", i, valid_rxns_retron, "\nRESN_USED", i, used_retrons)
            print("RESN_NOTUSED:", i, not_used)
        final_synthons[i] = set(not_used).union(not_reacting_cmds)
        cmd_to_cut = list(step_results.keys())
        for cmd in step_results:
            thisRetrons = step_results[cmd]['retrons']
            thisPairs = step_results[cmd]['pairs']
            thisRxes = step_results[cmd]['rxns']
            for onerx in thisRxes:
                left, right = onerx.split('>>')
                thisRet = left
                thisPair = tuple(right.split('.'))
                if thisRet not in retronToDirectSbses:
                    retronToDirectSbses[left] = set()
                #allDoneRxes[left].add(right)
                #for thisRet in thisRetrons:
                #    if thisRet not in retronToDirectSbses:
                #    retronToDirectSbses[thisRet] = set()
                retronToDirectSbses[thisRet].add(thisPair)
        if get_intermediates:
            intermediates.update(set(cmd_to_cut))
        nSteps += 1
        if not cmd_to_cut:
            break
    if debug:
        print(f"{nSteps} of {max_steps} steps, {final_synthons} ")
    if get_intermediates:
        return final_synthons, nSteps, retronToDirectSbses, intermediates
    return final_synthons, nSteps, retronToDirectSbses


def identify_Starting_Materials(set_of_synthons, mol_dict):
    staring_materials = set()
    boroacid1 = rxn_mols_prep.sanitize_mol('CN1C(C)(C)C(=O)OB(OC(=O)C1(C)C)C(C)(C)O')
    boroacid2 = rxn_mols_prep.sanitize_mol('CN1C(C)(C)C(=O)OB(OC(=O)C1(C)C)c1ccc(COC(O)=O)cc1')
    amino_boro_mol = Chem.MolFromSmarts('[NX3]C(=O)OCc1ccc(cc1)B([OH])[OH]')
    nbn_mol = Chem.MolFromSmarts('[n]c([B,B-])[n]')
    for synthon in set_of_synthons:
        check = Chem.MolFromSmiles(synthon).HasSubstructMatch(nbn_mol)
        check_prot_group = Chem.MolFromSmiles(synthon).HasSubstructMatch(amino_boro_mol)
        # print(check_prot_group)
        double_hal = inc_and_sel.identify_double_halogene(synthon)
        if synthon not in mol_dict.keys() or mol_dict[synthon]['pairs'] == []:
            # print('SYNTHON ',synthon)
            if synthon != boroacid1 and synthon != 'O=C=O' and synthon != 'CN(C(C)(C)C(=O)O)C(C)(C)C(=O)O' and synthon != boroacid2:
                if check is False and double_hal is False and check_prot_group is False:
                    staring_materials.add(synthon)
        # exceptions , add to the list syntons that can only be deprotected as starting materials
        if not check and double_hal is False and synthon != boroacid1 and synthon != boroacid2 and not check_prot_group and synthon in mol_dict:
            if mol_dict[synthon]['rxns_no'] == {'10'}:
                staring_materials.add(synthon)
            if mol_dict[synthon]['rxns_no'] == {'8'}:
                staring_materials.add(synthon)
            if mol_dict[synthon]['rxns_no'] == {'10', '8'}:
                staring_materials.add(synthon)
            if mol_dict[synthon]['rxns_no'] == {'13'}:
                staring_materials.add(synthon)
            if mol_dict[synthon]['rxns_no'] == {'13', '8'}:
                staring_materials.add(synthon)
    return staring_materials


def extract_list_from_dict(dictionary):
    res = []
    for k, v in dictionary.items():
        for el in v:
            res.append(Chem.MolFromSmiles(el))
    return res


def detect_fgs(smiles, smarts):
    res = Chem.MolFromSmiles(smiles).HasSubstructMatch(Chem.MolFromSmarts(smarts))
    return res


def count_fgs(smiles, smarts):
    res = Chem.MolFromSmiles(smiles).GetSubstructMatches(Chem.MolFromSmarts(smarts))
    return len(res)


def validate_synthons(synthons):
    bmida_groups = {'[#6][B]2OC(=O)C([CH3])([CH3])[N]([CH3])C([CH3])([CH3])C(=O)O2'}
    functional_groups = {'[#6]C(=O)[OH]', 'cBr', '[CX4][NH][CX4]', '[CX4,c,OX2][NH2]', '[CX3]=[CX3]Br'}
    deprotected = {'[BX3]([OH])[OH]'}
    special = {'[17N]'}
    res = []
    for synthon in synthons:
        c_bmida = 0
        c_fg = 0
        c_dep = 0
        c_special = 0

        for group in bmida_groups:
            check = count_fgs(synthon, group)
            c_bmida = c_bmida + check

        for fg in functional_groups:
            check_fg = count_fgs(synthon, fg)
            c_fg = c_fg + check_fg

        for group in deprotected:
            check_dep = count_fgs(synthon, group)
            c_dep = c_dep + check_dep

        for group in special:
            check_special = count_fgs(synthon, group)
            c_special = c_special + check_special

        c_ring_systems = inc_and_sel.GetRingSystems(Chem.MolFromSmiles(synthon))
        res.append(tuple([synthon, c_bmida, c_fg, c_dep, c_ring_systems, c_special]))

    final_mols = set()
    res1 = set()
    res2 = set()
    res3 = set()
    res4 = set()
    res5 = set()
    for r in res:
        if r[4] < 3:  # and r[5] == 0:
            if r[3] == 0:  # if there is no bornic acid
                if (r[1] == 0 and r[2] == 1) or (r[1] == 1 and r[2] == 0):
                    # print('Monofunctional.svg', r[0])
                    # 1x bmida or c_fg (acid, amine, Br)
                    final_mols.add(r[0])
                    res1.add(r[0])
                elif r[1] == 1 and r[2] > 0:
                    # print('Difunctional.svg', r[0])
                    # 1x bmida and 1+ (acid, amine, Br)
                    final_mols.add(r[0])
                    res2.add(r[0])
                else:
                    # print('Invalid.svg', r[0])
                    res3.add(r[0])
            else:
                # print('Extra group.svg', r[0])
                res4.add(r[0])
        else:
            # print('More than two ring systems.svg',r[0])
            res5.add(r[0])
    return res1, res2, res3, res4, res5, final_mols


def perform_protection_singlemol(mol, reactions, debug):
    changed = False
    for rxn in reactions:
        if debug:
            print("molprot:", mol, rxn['rxid'], rxn['rxname'])
        sbs = [mol, ] + rxn['fixedmols']
        newprod = None
        # inc_and_sel.identify_incomatibilities() #?? should we check it??
        while True:
            try:
                prods = rxn['retrorx'].RunReactants(sbs)
            except:
                print(f"PROBLEM w/ rx {AllChem.ReactionToSmarts(rxn['retrorx'])} with sbs: {[Chem.MolToSmiles(s) for s in sbs]}")
                raise
            if not prods:
                break
            newprod = prods[0][0]
            Chem.SanitizeMol(newprod)
            changed = True
            sbs = [newprod, ] + rxn['fixedmols']
        if newprod:
            mol = newprod
    return mol, changed


def perform_protection(compounds, reactions, origins, cmd_status, debug):
    new_list = []
    for cmd in compounds:
        mol = Chem.MolFromSmiles(cmd)
        mol, changed = perform_protection_singlemol(mol, reactions, debug)
        if changed:
            newcmd = Chem.MolToSmiles(mol)
            if origins:
                if newcmd not in origins:
                    origins[newcmd] = set()
                origins[newcmd].update(origins[cmd])
            if cmd_status:
                if newcmd not in cmd_status:
                    cmd_status[newcmd] = set()
                cmd_status[newcmd].update(cmd_status[cmd])
            cmd = newcmd
        new_list.append(cmd)
    return new_list, origins, cmd_status


def updateRetroToDirectSbs(retronToDirectSubstrates, retronToDirectSbs):
    # print('bigblocks:::W:',retronToDirectSubstrates, retronToDirectSbs)
    for retron in retronToDirectSbs:
        if retron not in retronToDirectSubstrates:
            retronToDirectSubstrates[retron] = retronToDirectSbs[retron]
        else:
            retronToDirectSubstrates[retron].update(retronToDirectSbs[retron])
    return retronToDirectSubstrates


def convertRetro2SbsToSbs2Retron(retronToDirectSubstrates):
    sbsToRetron = dict()
    for retron in retronToDirectSubstrates:
        for sbses in retronToDirectSubstrates[retron]:
            for sbs in sbses:
                if sbs not in sbsToRetron:
                    sbsToRetron[sbs] = set()
                sbsToRetron[sbs].add(retron)
    return sbsToRetron


def add_bigger_blocks(res_fg, removed_smi, sbsToRetronDict, origins, cmd_status, retrons_list, args):
    if args.debug:
        print("bigblocks: REMOVED", removed_smi)
    toadd = []
    status = dict()
    removed_set = set(removed_smi)
    retrons_set = set(retrons_list)
    sbsToRetron = sbsToRetronDict['sbs2retron']
    for rmsmi in removed_smi:
        if rmsmi in sbsToRetron:
            for smitoadd in sbsToRetron[rmsmi]:
                if smitoadd in retrons_set:
                    continue
                if smitoadd in res_fg:
                    continue
                addThisOne = True
                for pairs in sbsToRetronDict['retron2sbs'][smitoadd]:
                    isPairElemValid = [p not in removed_set for p in pairs]
                    print(f'bigblocks:: {smitoadd} == {pairs} {isPairElemValid}')
                    if all(isPairElemValid):
                        addThisOne = False
                        break
                if not addThisOne:
                    continue

                toadd.append(smitoadd)
                status[smitoadd] = cmd_status[rmsmi]
                if args.debug:
                    print(f"bigblocks: {smitoadd} FROM {rmsmi} values: {sbsToRetron[rmsmi]}")
    newdict = count_fg(toadd)
    # add filtration
    newdict, removed_bigger = filter_out(newdict, args)
    res_fg.update(newdict)
    if args.debug:
        print("REF added big:", res_fg.keys())
    for smi in newdict:
        if smi not in sbsToRetron:
            print("SMILES CANNOT BE CUT", smi)
            continue
        if smi not in origins:
            origins[smi] = sbsToRetron[smi].copy()
        else:
            origins[smi].update(sbsToRetron[smi])
        if smi not in cmd_status:
            cmd_status[smi] = status[smi]
        else:
            cmd_status[smi].add(status[smi])
    return res_fg, origins, cmd_status


# For each retron from the list extract synthons and add it to a dictionary where a key is a retron and values are either synthons
# or starting materials of a given retron
def _make_retroanalysis(rxn_list, inco_info, drug_info, args, drug_si, retros_list):
    if len(retros_list) != 1:
        raise NotImplementedError
    mol_dict = dict()
    # final_synthons = {}
    synthons_all = {}
    # retros_names = [drug_info.get(smi, smi) for smi in retros_list]
    retros_names = loader.getNames(retros_list, drug_info, drug_si)
    retros_list, _, _ = perform_protection(retros_list, rxn_list['init_protection'], None, None, args.debug)
    if args.debug:
        print("after protection:", retros_list)
    synthons_set = set()
    numberOfSteps = []
    origins = dict()
    cmd_status = dict()  #std, ext, extNF
    # cutinfo = dict()
    retronToDirectSubstrates = dict()
    for poz, r in enumerate(retros_list):
        if args.debug:
            print("RETROanaliz for", poz, r)
        try:
            res, nSteps, retronToDirectSbs = perform_Retro(rxn_list['retro'], rxn_list['protection'], [r, ], args.debug,
                                                           max_steps=args.maxgen, timeout=args.timeout)
            retronToDirectSubstrates = updateRetroToDirectSbs(retronToDirectSubstrates, retronToDirectSbs)
        except:
            print("RETRO FAILED FOR", r, traceback.format_exc())
            synthons_all[r] = {}
            continue
        numberOfSteps.append(nSteps)
        if args.debug:
            print("585::RES", r, "==>", res, mol_dict)
        if not res:
            synthons_all[r] = {}
            continue
        res_full = set.union(*res.values())
        for smi in res_full:
            if smi not in cmd_status:
                cmd_status[smi] = set()
            cmd_status[smi].add('std')
        if 'retro2' in rxn_list:
            try:
                res2, nSteps, retronToDirectSbs, all_intermediates = perform_Retro(rxn_list['retro2'], rxn_list['protection'],
                                                                                   tuple(res_full), args.debug, max_steps=args.maxgen,
                                                                                   get_intermediates=True, timeout=args.timeout)
                retronToDirectSubstrates = updateRetroToDirectSbs(retronToDirectSubstrates, retronToDirectSbs)
            except:
                print("RETRO FAILED FOR", r, traceback.format_exc())
                synthons_all[r] = {}
                continue
            all_products = set.union(*res2.values())
            if args.debug:
                print("598::RES", res_full, "===>", res2)
            all_final = all_products - all_intermediates
            all_notfinal = all_intermediates - all_final
            for smi in all_notfinal:
                if smi not in cmd_status:
                    cmd_status[smi] = set()
                cmd_status[smi].add('extNF')
            for smi in all_final:
                if smi not in cmd_status:
                    cmd_status[smi] = set()
                cmd_status[smi].add('ext')
            all_products.update(all_intermediates)
            if len(all_products) != (len(all_final) + len(all_notfinal)):
                print("ALL", len(all_products), "FINAL", len(all_final), "NF", len(all_notfinal))
                print(f"res {res2} imd: {all_intermediates}  all: {all_products} final: {all_final} NF: {all_notfinal}")
                raise
            res_full.update(all_products)
        # remove 0 generation
        if res.get(0, False):
            res_full = res_full - res[0]
        # stats number of product from target
        # if r in cutinfo:
        #    print(f"DUPLICATE!! {r}")
        #    if cutinfo[r] != res_full:
        #        print("XX", cutinfo[r], "RRRR", res_full)
        #        # raise NotImplementedError
        # cutinfo[r] = res_full

        for smi in res_full:
            if smi not in origins:
                origins[smi] = set()
            try:
                origins[smi].add(retros_names[poz])
            except IndexError:
                origins[smi].add(r)
        synthons_all[r] = res_full
        synthons_set.update(res_full)
    synthons_dict = count_fg(synthons_set)

    # filtration
    directSbsToRetron = convertRetro2SbsToSbs2Retron(retronToDirectSubstrates)
    directSbsToRetron = {'sbs2retron': directSbsToRetron, 'retron2sbs': retronToDirectSubstrates}
    res_fg, removed_smi = filter_out(synthons_dict, args)
    if args.checkParent == 'y' and removed_smi:
        res_fg, origins, cmd_status = add_bigger_blocks(res_fg, removed_smi, directSbsToRetron, origins, cmd_status, retros_list, args)
    if args.debug:
        print("SY", synthons_dict)
    mono = {smi: synthons_dict[smi] for smi in res_fg if _count_uniq_fg(synthons_dict[smi]) == 1}
    di = {smi: synthons_dict[smi] for smi in res_fg if _count_uniq_fg(synthons_dict[smi]) == 2}
    other = {smi: synthons_dict[smi] for smi in res_fg if not(smi in set(mono) or smi in set(di))}

    if args.swap != 'none':
        swapped_mono, origins, cmd_status = additional_synthons.swap_mono(mono, origins, cmd_status)
        swapped_mono = [smi for smi in swapped_mono if smi not in black_list]
        new_mono_info = count_fg(swapped_mono)
        swapped_mono, removed_smiles1 = filter_out(new_mono_info, args)
        if args.checkParent == 'y' and removed_smiles1:
            swapped_mono, origins, cmd_status = add_bigger_blocks(swapped_mono, removed_smiles1, directSbsToRetron, origins, cmd_status, retros_list, args)
        if args.debug:
            print("swapped_mono", swapped_mono)
        swapped_di, origins, cmd_status = additional_synthons.swap(di, origins, cmd_status)
        swapped_di = [smi for smi in swapped_di if smi not in black_list]
        new_di_info = count_fg(swapped_di)
        swapped_di, removed_smiles2 = filter_out(new_di_info, args)
        if args.checkParent == 'y' and removed_smiles2:
            swapped_di, origins, cmd_status = add_bigger_blocks(swapped_di, removed_smiles2, directSbsToRetron, origins, cmd_status, retros_list, args)
        if args.debug:
            print("swapped_di", swapped_di)
        all_mono = set(mono).union(set(swapped_mono))
        all_di = set(di).union(set(swapped_di))
    elif args.swap == 'none':
        all_mono = set(mono)
        all_di = set(di)
    else:
        raise NotImplementedError
    # multiply
    all_cmds = set.union(all_mono, all_di, set(other))
    all_new_cmds, origins, cmd_status = additional_synthons.multiply_synthons(all_cmds, rxn_list['multiply'], origins, cmd_status, args)
    # protection
    new_smiles, origins, cmd_status = perform_protection(all_new_cmds, rxn_list['protection'], origins, cmd_status, args.debug)
    new_synthons = count_fg(new_smiles)
    synthons_dict.update(new_synthons)
    # filtration here
    res_fg, removed_smiles = filter_out(new_synthons, args)
    if args.checkParent == 'y' and removed_smiles:
        res_fg, origins, cmd_status = add_bigger_blocks(res_fg, removed_smiles, directSbsToRetron, origins, cmd_status, retros_list, args)
    # new cmd categorize to mono, di, other
    new_mono = {smi: synthons_dict[smi] for smi in res_fg if _count_uniq_fg(synthons_dict[smi]) == 1}
    new_di = {smi: synthons_dict[smi] for smi in res_fg if _count_uniq_fg(synthons_dict[smi]) == 2}
    all_mono = tuple(set(new_mono.keys()).union(all_mono))
    all_di = tuple(set(new_di.keys()).union(all_di))
    new_other = {smi: synthons_dict[smi] for smi in res_fg if not(smi in set(all_mono) or smi in set(all_di))}
    all_other = tuple(set(new_other.keys()).union(other))
    if args.debug:
        print("ALL_MONO:580:", '.'.join(all_mono))
    # warning and protection
    all_mono, origins, cmd_status = perform_replace(all_mono, origins, cmd_status, args)
    all_di, origins, cmd_status = perform_replace(all_di, origins, cmd_status, args)

    if args.debug:
        print("ALLMONO", all_mono, "ALLDI", all_di)
    all_mono, warnings_mono, warning_info1, origins, cmd_status = get_warnings_and_protect(all_mono, inco_info, rxn_list['final_protection'],
                                                                                           origins, cmd_status, args)
    if args.debug:
        print("MONO", all_mono, warnings_mono, warning_info1)
    all_di, warnings_di, warning_info2, origins, cmd_status = get_warnings_and_protect(all_di, inco_info, rxn_list['final_protection'],
                                                                                       origins, cmd_status, args)
    if args.debug:
        print("DI", all_di, warnings_di, warning_info2)
    mono_deprot_final, map_dict1 = perform_deprotection(all_mono, res_fg, rxn_list['deprotection'])
    di_deprot_final, map_dict2 = perform_deprotection(all_di, res_fg, rxn_list['deprotection'])
    # mono_deprot_final = perform_replace(mono_deprotected.keys())
    # di_deprot_final = perform_replace(di_deprotected.keys())
    # all_mono = perform_replace(all_mono)
    # all_di = perform_replace(all_di)
    dct = {'mono': set(all_mono), 'di': set(all_di), 'mono_deprotected': set(mono_deprot_final),
           'di_deprotected': set(di_deprot_final), 'other': set(all_other)}
    warnings_info_final = _form_warnings([warning_info1, warning_info2], [map_dict1, map_dict2])
    origins = {'target': origins, 'status': cmd_status}
    return synthons_all, dct, warnings_info_final, numberOfSteps, origins, retronToDirectSubstrates


def make_retroanalysis(retros_list, rxn_list, inco_info, drug_info, killers, args, drug_si=None):
    subproc_function = partial(_make_retroanalysis, rxn_list, inco_info, drug_info, args, drug_si)

    def _subprocess(qin, qout):
        while True:
            msg = qin.get()  # [smiles, smiles] or 'STOP'
            if msg == 'STOP':
                break
            if msg[0] in killers:
                print("IGNORE", msg[0], file=sys.stderr)
                res = 'IGNORE'
            else:
                res = subproc_function(msg)
            qout.put((msg, res))
        return None

    qin, qout, = Queue(), Queue()
    num_procs = min(len(retros_list), args.numprocs)
    procs = [Process(target=_subprocess, args=(qin, qout)) for _ in range(num_procs)]
    _ = [p.start() for p in procs]
    msgtoget = 0
    cutinfo = dict()
    f_synthons_all = dict()
    f_dct = {'mono': set(), 'di': set(), 'mono_deprotected': set(), 'di_deprotected': set(), 'other': set()}
    warnings, f_numberOfStep = dict(), []
    f_origins = {'target': dict(), 'status': dict()}
    f_retronToSbs = dict()
    submited = set()
    for smi in retros_list:
        qin.put([smi, ])
        msgtoget += 1
        submited.add(smi)
    sanityCheck = dict()
    for idx in range(msgtoget):
        tries = 0
        while True:
            tries += 1
            try:
                smiles_list, raw_res = qout.get(timeout=30)
                if smiles_list[0] in submited:
                    submited.remove(smiles_list[0])
                else:
                    print("NOT FOUND IN SUBMMITTED", smiles_list)
                break
            except:
                print("LEN to get::", len(submited))
                if len(submited) < 10:
                    print(f"PROBLEMATIC:toget: {len(submited)} :: {'.'.join(submited)} banned {len(smiles_killer)}", file=sys.stderr)
                if args.timeout:
                    timenow = tries / 2
                    if timenow > args.timeout:
                        raise NotImplementedError
        if raw_res == 'IGNORE':
            continue
        synthons_all, dct, warning_info, numberOfStep, origins, retronToSbs = raw_res
        assert len(smiles_list) == 1
        smiles = smiles_list[0]
        if smiles in cutinfo:
            print("---DUPLICATE", smiles, "WAS", cutinfo[smiles], "IS::", set(dct['mono']).union(set(dct['di'])))
        # cutinfo[smiles] = set(dct['mono']).union(set(dct['di']))
        cutinfo[smiles] = dct.copy()
        if not numberOfStep:
            cutinfo[smiles]['numGens'] = 0
        else:
            assert len(numberOfStep) == 1
            cutinfo[smiles]['numGens'] = numberOfStep[0]
        cutinfo[smiles]['protected'] = list(synthons_all.keys())[0]
        # print(f"++++ {smiles} === {cutinfo[smiles]}")
        f_synthons_all, f_dct, warnings, f_numberOfStep, f_origins, f_retronToSbs = combine_results(raw_res, f_synthons_all, f_dct,
                                                                                                   warnings, f_numberOfStep, f_origins, f_retronToSbs)
        if smiles in sanityCheck:
            print("WAS", sanityCheck[smiles])
            print("NEW", dct)
        else:
            sanityCheck[smiles] = dct
        if args.progress:
            if idx < 0.75 * msgtoget:
                if idx % 25 == 0:
                    print(f"{idx} ", end=' ', file=sys.stderr, flush=True)
            else:
                print(f"{idx} ", end=' ', file=sys.stderr, flush=True)
    for smi in retros_list:
        qin.put('STOP')
    qin.close()
    qout.close()
    _ = [p.join() for p in procs]
    f_warning_info = _make_list_from_warnings(warnings)
    return f_synthons_all, f_dct, f_warning_info, f_numberOfStep, f_origins, cutinfo


def combine_results(raw_res, f_synthons_all, f_dct, warning_dict, f_numberOfStep, f_origins, f_retronToSbs):
    synthons_all, dct, warning_info, numberOfStep, origins, retronToSbs = raw_res
    f_synthons_all.update(synthons_all)
    for k in f_dct:
        f_dct[k].update(dct[k])
    #f_warning_info.extend(warning_info)
    f_numberOfStep.extend(f_numberOfStep)
    for smi in origins['target']:
        if smi not in f_origins['target']:
            f_origins['target'][smi] = origins['target'][smi]
        else:
            f_origins['target'][smi].update(origins['target'][smi])
    for smi in origins['status']:
        if smi not in f_origins['status']:
            f_origins['status'][smi] = origins['status'][smi]
        else:
            f_origins['status'][smi].update(f_origins['status'][smi])
    # f_origins['status'].update(origins['status'])
    # dct = {'mono': all_mono, 'di': all_di, 'mono_deprotected': tuple(mono_deprot_final),
    #       'di_deprotected': tuple(di_deprot_final), 'other': all_other}
    for lst in warning_info:
        if lst[0] not in warning_dict:
            warning_dict[lst[0]] = []
        warning_dict[lst[0]].append(lst[1:])
    for smi in retronToSbs:
        if smi not in f_retronToSbs:
            f_retronToSbs[smi] = retronToSbs[smi]
        else:
            f_retronToSbs[smi].update(retronToSbs[smi])
    return f_synthons_all, f_dct, warning_dict, f_numberOfStep, f_origins, f_retronToSbs


def _make_list_from_warnings(warning_dict):
    lst = []
    for k in warning_dict:
        if len(set(warning_dict[k])) == 1:
            thislst = [k, ] + list(warning_dict[k][0])
            lst.append(thislst)
        else:
            print("===", k, len(set(warning_dict[k])))
            raise NotImplementedError
    return lst


def _form_warnings(warnings_lists, map_dict_list):
    assert len(warnings_lists) == len(map_dict_list)
    full_warnings = []
    for num, warning_list_dict in enumerate(warnings_lists):
        for warning in warning_list_dict['ok']:
            smi, probl, sma = warning
            new_smi_list = set(map_dict_list[num][smi])
            info = (smi, '.'.join(new_smi_list), probl, sma)
            full_warnings.append(info)
        for warning in warning_list_dict['problem']:
            smi, probl, sma = warning
            new_smi_list = set(map_dict_list[num][smi])
            info = (smi, '.'.join(new_smi_list), probl, sma)
            full_warnings.append(info)
    return full_warnings


def _get_cores(mol):
    info = dict()
    core = set()
    for fgname in fg_mols:
        matches = mol.GetSubstructMatches(fg_mols[fgname])
        if matches:
            if fgname == 'bmida':
                cbz_match = mol.GetSubstructMatches(cbzmol)
                if len(cbz_match):
                    continue
            info[fgname] = matches
            for match in matches:
                core.update(set(match))
    return info, core


def _make_final_protection(mol, inco_mol, matches, matches_to_protect, rxn_list, debug):
    # initial version here should be more logic to handle different cases and reaction type
    # but now for and only one reaction it is good enough
    changed = False
    matches_sets = [set(match) for match in matches]
    not_allowed_sets = [set(match) for match in matches_to_protect]
    for rx in rxn_list:
        protection_matches = mol.GetSubstructMatches(rx['core'])
        if not protection_matches:
            # core of protecting reaction not found go to next protecting reaction
            continue
        if debug:
            print("RX", rx['rxid'], "protMatch", protection_matches, "Match", matches, "notallow",
                  matches_to_protect, "INCO", Chem.MolToSmarts(inco_mol))
        active_cores = []
        for prot_match in protection_matches:
            prot_match_set = set(prot_match)
            if any([matchset.intersection(prot_match_set) for matchset in matches_sets]):
                # print("INTERSEC", matches_sets, prot_match_set)
                if any([notallow.intersection(prot_match_set) for notallow in not_allowed_sets]):
                    # continue
                    if debug:
                        print("==", matches_to_protect, prot_match)
                active_cores.append(prot_match)
        # active_cores = protection_matches
        if debug:  # and active_cores:
            print("RXN", rx['rxid'], "act cores", active_cores, "prot", protection_matches, "notAllow", not_allowed_sets)
        if active_cores:
            if len(active_cores) != len(protection_matches):
                print("some of cores are not allowed for rx", Chem.MolToSmiles(mol))
                # raise NotImplementedError
            if debug:
                print("BEFORE FPMOL", Chem.MolToSmiles(mol))
            mol, changed = perform_protection_singlemol(mol, [rx, ], debug)
            if debug:
                print("AFTER FPMOL", Chem.MolToSmiles(mol))
    return mol, changed


def _append_to_problems(problems, mol, smi, cores, inco_dict, prot_rxn_list, debug):
    if debug:
        print("==BEGPRO", smi, cores)
    for poz, sma_mol in enumerate(inco_dict['mols']):
        matches = mol.GetSubstructMatches(sma_mol)
        _, cores = _get_cores(mol)
        if not matches:
            # no group to protect found - dont do protection
            continue
        matches_to_protect = [match for match in matches if not len(set(match).intersection(cores)) > 1]
        # matches_to_protect = []  # allow protection everywhere
        if debug:
            print("match to protect", matches_to_protect)
        new_mol, is_protected = _make_final_protection(mol, sma_mol, matches, matches_to_protect, prot_rxn_list, debug)
        if debug:
            print("FINAL PROT", smi, "==>", Chem.MolToSmiles(new_mol), is_protected, matches_to_protect)
            print("CORES", cores, matches, Chem.MolToSmarts(sma_mol))
            print("PROT RX LIST", [rx['rxid'] for rx in prot_rxn_list])
        if matches_to_protect and not is_protected:
            if smi not in problems:
                problems[smi] = []
            problems[smi].append(inco_dict['smarts'][poz])
        if is_protected:
            oldsmi = smi
            smi = Chem.MolToSmiles(new_mol)
            # _, cores = _get_cores(new_mol)
            mol = new_mol
            if oldsmi in problems:
                problems[smi] = problems[oldsmi]
                del problems[oldsmi]
    if debug:
        print("==ENDPRO", smi, cores)
    return problems, smi, cores


def get_warnings_and_protect(smi_list, inco_info, prot_rxn_list, origins, cmd_status, args):
    # to_all_check = {'bmida', 'aromBr', 'secN', 'primN', 'vinylBr', 'phenols'}
    problems = dict()
    new_smi_list = []
    for smi in smi_list:
        oryg = smi
        mol = Chem.MolFromSmiles(smi)
        sminew = None
        found_FG, cores = _get_cores(mol)
        if args.debug:
            print("ALL FG", found_FG, smi)
        if len(found_FG) > 1:
            # check for all
            problems, sminew, cores = _append_to_problems(problems, mol, smi, cores, inco_info['ALL'], prot_rxn_list, args.debug)
        elif len(found_FG) == 1:
            the_fg = list(found_FG.keys())[0]
            if the_fg in {'bmida', 'secN', 'primN', 'hydroxyamine'}:
                problems, sminew, cores = _append_to_problems(problems, mol, smi, cores, inco_info['MIDDLE'], prot_rxn_list, args.debug)
                # MIDDLE := Suzuki Buchwald 'Amide synthesis' 'Coupling of carboxylic acid with hydrazine'
                #                         'BTIDA hydrolysis'   'N-Cbz-BTIDA deprotection'
            elif the_fg == 'phenols':
                problems, sminew, cores = _append_to_problems(problems, mol, smi, cores, inco_info['MIDDLE'], prot_rxn_list, args.debug)
                if smi != sminew:
                    smi = sminew
                    mol = Chem.MolFromSmiles(sminew)
                # MIDDLE + Pd CO coupling
                inc_part = inco_info['Pd-coupling of aromatic bromide with phenol']
                problems, sminew, cores = _append_to_problems(problems, mol, smi, cores, inc_part, prot_rxn_list, args.debug)
                if smi != sminew:
                    smi = sminew
                    mol = Chem.MolFromSmiles(sminew)
            elif the_fg == 'heteroaroN':
                problems, sminew, cores = _append_to_problems(problems, mol, smi, cores, inco_info['MIDDLE'], prot_rxn_list, args.debug)
                if smi != sminew:
                    smi = sminew
                    mol = Chem.MolFromSmiles(sminew)
                problems, sminew, cores = _append_to_problems(problems, mol, smi, cores, inco_info['nH coupling'], prot_rxn_list, args.debug)
                # MIDDLE + nH coupling
            elif the_fg == 'vinylBr' or the_fg == 'CBr':
                problems, sminew, cores = _append_to_problems(problems, mol, smi, cores, inco_info['Suzuki'], prot_rxn_list, args.debug)
            elif the_fg == 'aromBr':
                problems, sminew, cores = _append_to_problems(problems, mol, smi, cores, inco_info['Suzuki'], prot_rxn_list, args.debug)
                if smi != sminew:
                    smi = sminew
                    mol = Chem.MolFromSmiles(sminew)
                problems, sminew, cores = _append_to_problems(problems, mol, smi, cores, inco_info['Buchwald'], prot_rxn_list, args.debug)
                if smi != sminew:
                    smi = sminew
                    mol = Chem.MolFromSmiles(sminew)
                if args.warning == 'full':
                    inc_part = inco_info['Pd-coupling of aromatic bromide with phenol']
                    problems, sminew, cores = _append_to_problems(problems, mol, smi, cores, inc_part, prot_rxn_list, args.debug)
                    if smi != sminew:
                        smi = sminew
                        mol = Chem.MolFromSmiles(sminew)
                    problems, sminew, cores = _append_to_problems(problems, mol, smi, cores, inco_info['nH coupling'],
                                                                  prot_rxn_list, args.debug)
            elif the_fg == 'COOH':
                problems, sminew, cores = _append_to_problems(problems, mol, smi, cores, inco_info['Amide synthesis'],
                                                              prot_rxn_list, args.debug)
            elif the_fg == 'O=C=N':
                problems, sminew, cores = _append_to_problems(problems, mol, smi, cores, inco_info['Addition of amines to isocyanates'],
                                                              prot_rxn_list, args.debug)
        if sminew:
            new_smi_list.append(sminew)
            if sminew not in origins:
                origins[sminew] = set()
            origins[sminew].update(origins[smi])
            if sminew not in cmd_status:
                cmd_status[sminew] = set()
            cmd_status[sminew].update(cmd_status[smi])
        else:
            new_smi_list.append(smi)
        if args.debug and sminew != oryg:
            print("MODED", oryg, sminew)
    warning_info = {'problem': [], 'ok': []}
    # if debug:
    # fh = open('wyniki_niekompatybilnosci.csv', 'a')
    for smi in sorted(problems):
        sma_orygin = _get_smarts_orygin(problems[smi], inco_info)
        # print(smi, '.'.join(problems[smi]), sma_orygin, sep=';', file=fh)
        warning_info['problem'].append((smi, '.'.join(problems[smi]), sma_orygin))
    for smi in sorted(new_smi_list):
        if smi not in problems:
            # print(smi, '_NO_PROBLEM_', sep=';', file=fh)
            warning_info['problem'].append((smi, '_NO_PROBLEM_', ''))
    #    fh.close()
    return new_smi_list, problems, warning_info, origins, cmd_status


def _get_smarts_orygin(sma_list, inco_info):
    origins = []
    for sma in sma_list:
        found = []
        for rx in inco_info:
            if rx == 'ALL':
                continue
            if sma in set(inco_info[rx]['smarts']):
                rxname = rx.replace(' ', '_')
                found.append(rxname)
        if found:
            origins.append('__OR__'.join(found))
    return '.'.join(origins)


aroHalo = Chem.MolFromSmarts('[$([Cl,I,At][c])]')
metyl = Chem.MolFromSmiles('[1CH3]')
def perform_replace(smi_list, origins, cmd_status, args):
    # make on smarts level
    if args.replaceMode == 'methylAll':
        new_list = []  # Chem.CanonSmiles(smi.replace('[At]', '[4CH3]')) for smi in smi_list]
        for smi in smi_list:
            smiles = smi.replace('[At]', '[4CH3]')
            mol = Chem.MolFromSmiles(smiles)
            prods = AllChem.ReplaceSubstructs(mol, aroHalo, metyl, replaceAll=True)
            new_smiles = Chem.MolToSmiles(prods[0])
            if new_smiles != smi:
                if new_smiles not in origins:
                    origins[new_smiles] = set()
                origins[new_smiles].update(origins[smi])
                if new_smiles not in cmd_status:
                    cmd_status[new_smiles] = set()
                cmd_status[new_smiles].update(cmd_status[smi])
            new_list.append(new_smiles)
    elif args.replaceMode == 'methylNoX':
        new_list = []  # Chem.CanonSmiles(smi.replace('[At]', '[4CH3]')) for smi in smi_list]
        for smi in smi_list:
            smiles = smi.replace('[At]', '[4CH3]')
            mol = Chem.MolFromSmiles(smiles)
            new_smiles = Chem.MolToSmiles(mol)
            if new_smiles != smi:
                if new_smiles not in origins:
                    origins[new_smiles] = set()
                origins[new_smiles].update(origins[smi])
                if new_smiles not in cmd_status:
                    cmd_status[new_smiles] = set()
                cmd_status[new_smiles].update(cmd_status[smi])
            new_list.append(new_smiles)
    elif args.replaceMode == 'noReplace':
        new_list = []  # Chem.CanonSmiles(smi.replace('[At]', '[4CH3]')) for smi in smi_list]
        for smi in smi_list:
            smiles = smi.replace('[At]', '[Br]')
            mol = Chem.MolFromSmiles(smiles)
            new_smiles = Chem.MolToSmiles(mol)
            if new_smiles != smi:
                if new_smiles not in origins:
                    origins[new_smiles] = set()
                origins[new_smiles].update(origins[smi])
                if new_smiles not in cmd_status:
                    cmd_status[new_smiles] = set()
                cmd_status[new_smiles].update(cmd_status[smi])
            new_list.append(new_smiles)
    elif args.replaceMode == 'none':
        return smi_list, origins, cmd_status
    else:
        raise NotImplementedError
    return new_list, origins, cmd_status


def _count_uniq_fg(smidict):
    # fg_names = ['bmida', 'COOH', 'aromBr', 'secN', 'primN', 'vinylBr']
    uniqfg = 0
    if smidict['bmida'] > (smidict['secN'] + smidict['primN']):
        uniqfg += 1
    if smidict['COOH'] > 0:
        uniqfg += 1
    if smidict['aromBr'] + smidict['vinylBr'] + smidict['CBr'] > 0:
        uniqfg += 1
    if smidict['secN'] + smidict['primN'] + smidict['hydroxyamine'] + smidict['heteroaroN'] > 0:
        uniqfg += 1
    if smidict['phenols'] > 0:
        uniqfg += 1
    if smidict['O=C=N'] > 0:
        uniqfg += 1
    return uniqfg


def count_fg(smiles_list):
    # bmida, primary N, secondary N, acid, -Br
    dct = dict()
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        info = {fg: len(mol.GetSubstructMatches(fg_mols[fg])) for fg in fg_mols}
        dct[smi] = info
        dct[smi]['rings'] = mol.GetRingInfo().NumRings()
        dct[smi]['C'] = count_carbons(smi)
    return dct


def getRetrons(smiList):
    # rxn_list = load_reactions()
    print("RXNLIST", len(rxn_list), "SMILIT", len(smiList))
    _, res, _ = make_retroanalysis(smiList, rxn_list)
    # print("RESRAW", _)
    print("RESULT", len(res))
    return list(res)


def deduplicate_lists(list1, list2):
    res = []
    tmp = []
    for e in list1:
        if e not in list2:
            tmp.append(list1)
    for e in tmp:
        if e not in res:
            res.append(e)
    return res


hydroxyamine_patt = Chem.MolFromSmarts('[NX3H2,$([NX3H1][CX4,c])]O')
#CdoubleC = Chem.MolFromSmarts('[#6][CX3]=[CX3][#6]')
CdoubleC = Chem.MolFromSmarts('[CX3]=[CX3]')
CtripleC = Chem.MolFromSmarts('[C]#[C]')
def filter_out(res_fg, args):
    torm = []
    for smi in res_fg:
        if smi == 'CN':
            if args.debug:
                print("remove metylamine")
            torm.append(smi)
            continue
        if smi in black_list:
            if args.debug:
                print("remove due black_list", smi)
            torm.append(smi)
            continue
        # no 'nc([B,B-])n'
        diazoBoron = None
        if args.diazodef == 'general':
            diazoBoron = res_fg[smi]['diazoBoronGeneral']
        elif args.diazodef == 'r6':
            diazoBoron = res_fg[smi]['diazoBoronR6']
        else:
            print("DIAZODEF", args.diazodef)
            raise NotImplementedError
        if diazoBoron > 0:
            if args.debug:
                print("removed:diazo", smi)
            torm.append(smi)
            continue
        totalN = res_fg[smi]['primN'] + res_fg[smi]['secN']
        # max one amine
        if totalN > 1:
            torm.append(smi)
            if args.debug:
                print("removed: polyamine", smi)
            continue
        # max 1 bmida
        if res_fg[smi]['bmida'] - totalN > 1:
            torm.append(smi)
            if args.debug:
                print("removed:2+bmida:", smi)
            continue
        # no B and N
        if totalN > 0 and res_fg[smi]['bmida'] - totalN > 0:
            if args.debug:
                print("removed:amine+boronic:", smi)
            torm.append(smi)
            continue
        totalBr = res_fg[smi]['vinylBr'] + res_fg[smi]['aromBr'] + res_fg[smi]['CBr']
        # max 1 cooh
        if res_fg[smi]['COOH'] > 1:
            torm.append(smi)
            if args.debug:
                print("removed:2+COOH:", smi)
            continue
        # max 1 O=C=N
        if res_fg[smi]['O=C=N'] > 1:
            torm.append(smi)
            if args.debug:
                print("removed:2+ O=C=N:", smi)
            continue
        # no O=C=N and other gr
        if res_fg[smi]['O=C=N'] and (totalN + res_fg[smi]['bmida'] + res_fg[smi]['COOH'] + totalBr + res_fg[smi]['phenols']) > 0:
            torm.append(smi)
            if args.debug:
                print("removed O=C=N with second FG", smi)
            continue
        # no COOH with Br
        if res_fg[smi]['COOH'] == 1 and totalBr > 0:
            torm.append(smi)
            if args.debug:
                print("removed:COOH + Br:", smi)
            continue
        # max 1 Br
        if totalBr > 1:
            torm.append(smi)
            if args.debug:
                print("removed:poly Br:", smi)
            continue
        # if totalBr == 1:
        #    # totalHalo = res_fg[smi]['vinylHalo'] + res_fg[smi]['aromHalo']
        #    # if totalHalo > 0:
        #    #    if args.debug:
        #    #        print("removed:Br + halo:", smi)
        #    #    torm.append(smi)
        #    #    continue
        #    if totalN > 0:
        #        if args.debug:
        #            print("removed: N Br:", smi)
        #        torm.append(smi)
        #        continue
        # no poly fenols
        if res_fg[smi]['phenols'] > 1:
            torm.append(smi)
            if args.debug:
                print("removed:2+fenols:", smi)
            continue
        # no fenols with other FG
        if res_fg[smi]['phenols'] == 1:
            if (totalBr + totalN + res_fg[smi]['bmida']) > 0:
                torm.append(smi)
                if args.debug:
                    print("removed:fenol+FG:", smi)
                continue
            # check aga motifs
            if _has_forbidden_phenols(smi):
                torm.append(smi)
                if args.debug:
                    print("removed:fenol with Aga's motif:", smi)
                continue
        # no poly nH
        if res_fg[smi]['heteroaroN'] > 1:
            torm.append(smi)
            if args.debug:
                print("removed:2+ nH:", smi)
            continue
        if res_fg[smi]['heteroaroN'] == 1 and (totalBr + totalN + res_fg[smi]['bmida']) > 0:
            torm.append(smi)
            if args.debug:
                print("removed: nH +FG:", smi)
            continue
        rings = res_fg[smi]['rings'] - res_fg[smi]['secN'] - res_fg[smi]['primN']
        if rings > 6:
            torm.append(smi)
            if args.debug:
                print("removed:6+rings:", smi)
            continue
        # remove during printing on unprotected
        # if res_fg[smi]['C'] > args.maxC:
        #    torm.append(smi)
        #    if args.debug:
        #        print("removed due to num C", smi, "number of C", res_fg[smi]['C'], 'limit', args.maxC)
        #    continue
        # remove hydroxyamine
        # if 'N' in smi and 'O' in smi:
        #    mol = Chem.MolFromSmiles(smi)
        #    if mol.HasSubstructMatch(hydroxyamine_patt):
        #        torm.append(smi)
        #        if debug:
        #            print("remove hydroxylamine", smi)
        #        continue
        # remove C=C and C#C
        # if '=' in smi or '#' in smi:
        #    hasNfg = res_fg[smi]['primN'] or res_fg[smi]['secN'] or res_fg[smi]['phenols'] or res_fg[smi]['heteroaroN'] or res_fg[smi]['bmida']
        #    countFG = _count_uniq_fg(res_fg[smi])
        #    if countFG > 1 or hasNfg:
        #        mol = Chem.MolFromSmiles(smi)
        #        if mol.HasSubstructMatch(CtripleC) or mol.HasSubstructMatch(CdoubleC):
        #            torm.append(smi)
        #            if args.debug:
        #                print("remove due C=C/C#C", smi)
        #            continue
    for smi in torm:
        del res_fg[smi]
    return res_fg, torm


def perform_deprotection(smilist, res_fg, deprot_rxns):
    dct = dict()
    map_dict = dict()
    for smi in smilist:
        mol = None
        added = False
        if smi not in res_fg:
            res_fg.update(count_fg([smi, ]))
        if res_fg[smi]['bmida'] and (res_fg[smi]['primN'] + res_fg[smi]['secN']) == 0:
            rxn = deprot_rxns[0]
            if not mol:
                mol = Chem.MolFromSmiles(smi)
            sbs = [mol, ] + rxn['fixedmols']
            prods = rxn['retrorx'].RunReactants(sbs)
            newprod = prods[0][0]
            Chem.SanitizeMol(newprod)
            newsmi = Chem.MolToSmiles(newprod)
            dct[newsmi] = res_fg[smi]
            if smi not in map_dict:
                map_dict[smi] = []
            map_dict[smi].append(newsmi)
            added = True
        if res_fg[smi]['primN']:
            rxn = deprot_rxns[1]
            if not mol:
                mol = Chem.MolFromSmiles(smi)
            sbs = [mol, ] + rxn['fixedmols']
            for prods in rxn['retrorx'].RunReactants(sbs):
                newprod = prods[0]
                Chem.SanitizeMol(newprod)
                newsmi = Chem.MolToSmiles(newprod)
                dct[newsmi] = res_fg[smi]
                if smi not in map_dict:
                    map_dict[smi] = []
                map_dict[smi].append(newsmi)
                added = True
        if res_fg[smi]['secN']:
            rxn = deprot_rxns[2]
            if not mol:
                mol = Chem.MolFromSmiles(smi)
            sbs = [mol, ] + rxn['fixedmols']
            for prods in rxn['retrorx'].RunReactants(sbs):
                newprod = prods[0]
                Chem.SanitizeMol(newprod)
                newsmi = Chem.MolToSmiles(newprod)
                dct[newsmi] = res_fg[smi]
                if smi not in map_dict:
                    map_dict[smi] = []
                map_dict[smi].append(newsmi)
                added = True
        if not added:
            dct[smi] = res_fg[smi]
            if smi not in map_dict:
                map_dict[smi] = []
            map_dict[smi].append(smi)
    return dct, map_dict


def count_carbons(smi, exclude_protection=False):
    mol = Chem.MolFromSmiles(smi)
    return count_carbons_from_mol(mol, exclude_protection)


def count_carbons_from_mol(mol, exclude_protection):
    q = rdqueries.AtomNumEqualsQueryAtom(6)
    c_no = len(mol.GetAtomsMatchingQuery(q))
    substruct = Chem.MolFromSmarts('CC(C)(C)OC(=O)[NX3,n]')
    if mol.HasSubstructMatch(substruct):
        c_no = c_no - 5
    if mol.HasSubstructMatch(Chem.MolFromSmarts('[1CH3]')):
        c_no = c_no - 1
    if mol.HasSubstructMatch(Chem.MolFromSmarts('[2CH3]')):
        c_no = c_no - 1
    if mol.HasSubstructMatch(Chem.MolFromSmarts('[3CH3]')):
        c_no = c_no - 1
    if mol.HasSubstructMatch(Chem.MolFromSmarts('[4CH3]')):
        c_no = c_no - 1
    if mol.HasSubstructMatch(Chem.MolFromSmarts('CC(C)[Si]([CX2])(C(C)C)C(C)C')):
        c_no = c_no - 9
    if exclude_protection:
        if mol.HasSubstructMatch(cbzmol):
            c_no -= 8
        elif mol.HasSubstructMatch(fg_mols['bmida']):
            c_no -= 9
    return c_no


def add_results_from_file(fname):
    fh = open(fname)
    lines = fh.readlines()
    fh.close()
    results = {'mono': [], 'mono_deprot': [], 'di': [], 'di_deprot': [], 'lines': []}
    protected, deprotected = [], []
    for line in lines:
        line = line[:-1]
        elems = line.split(';')
        results['lines'].append(line)
        protected.append(elems[0])
        deprotected.append(elems[1])
    protected_dict = count_fg(protected)
    # filtration
    # res_fg = filter_out(synthons_dict, args)
    mono = set([smi for smi in protected if _count_uniq_fg(protected_dict[smi]) == 1])
    di = set([smi for smi in protected if _count_uniq_fg(protected_dict[smi]) == 2])
    for idx, prot_smi in enumerate(protected):
        if prot_smi in mono:
            assert prot_smi not in di
            results['mono'].append(prot_smi)
            results['mono_deprot'].append(deprotected[idx])
        elif prot_smi in di:
            assert prot_smi not in mono
            results['di'].append(prot_smi)
            results['di_deprot'].append(deprotected[idx])
        else:
            raise NotImplementedError
    return results


def print_results(results, warnings, origins, cutinfo, args):
    # print('all_mono', len(results['mono']), '.'.join(results['mono']))
    # print('all_di', len(results['di']), '.'.join(results['di']))
    print('other', len(results['other']), '.'.join(results['other']))
    # print('deprotected_mono', len(results['mono_deprotected']), '.'.join(results['mono_deprotected']))
    # print('deprotected_di', len(results['di_deprotected']), '.'.join(results['di_deprotected']))
    mono_set = set(results['mono'])
    di_set = set(results['di'])
    other_set = set(results['other'])
    mono_deprot_set = set(results['mono_deprotected'])
    di_deprot_set = set(results['di_deprotected'])
    fh = open(f'{args.output}.csv', 'w')
    new_mono = []
    new_di = []
    new_other = []
    new_mono_deprot = []
    new_di_deprot = []
    header = ['smiles_protected', 'smiles_deprotected', 'block_type', 'reaction_set_used', 'numC',
              'parent_targets', 'num_targets', 'problematic_groups', 'problematic_reactions']
    print(*header, sep=';', file=fh)
    if args.includeHardcodedCuts:
        aga_molecules = add_results_from_file(args.includeHardcodedCuts)
        new_mono.extend(aga_molecules['mono'])
        new_di.extend(aga_molecules['di'])
        new_mono_deprot.extend(aga_molecules['mono_deprot'])
        new_di_deprot.extend(aga_molecules['di_deprot'])
        for line in aga_molecules['lines']:
            print(line, file=fh)
    heavy_mono_di = set()
    for line in warnings:
        numC = count_carbons(line[1])
        if line[0] in origins['target']:
            drug = origins['target'][line[0]]
        elif line[1] in origins:
            drug = origins['target'][line[1]]
        else:
            drug = '???'
        cmd_status = 'WTF'
        if line[0] in origins['status']:
            cmd_status = '.'.join(sorted(origins['status'][line[0]]))
        elif line[1] in origins['status']:
            cmd_status = '.'.join(sorted(origins['status'][line[1]]))
        if line[1] == 'CN':   # remove methylamine - it is ok but too small
            continue
        if args.heavyMode == 'maxC':
            if numC > args.maxC:  # remove heavy
                if line[0] in mono_set or line[0] in di_set:
                    heavy_mono_di.add(line[0])
                continue
        elif args.heavyMode == '19':
            if numC > args.maxC:  # remove heavy defined as 19
                continue
            if numC > 19 and line[0] in mono_set or line[0] in di_set:
                heavy_mono_di.add(line[0])

        else:
            raise NotImplementedError
        fg_status = 'unknown'
        if line[0] in mono_set:
            fg_status = 'mono'
        elif line[0] in di_set:
            fg_status = 'di'
        print(line[0], line[1], fg_status, cmd_status, numC, '.'.join(drug), len(drug), *line[2:], sep=';', file=fh)
        # make new list
        if line[0] in mono_set:
            assert line[1] in mono_deprot_set
            if line[0] in di_set:
                print(f'WARNING: {line[0]} is in mono and di')
            assert line[0] not in other_set
            new_mono.append(line[0])
            if line[1] in di_deprot_set:
                print(f'WARNING: {line[1]} is in mono and di_deprot')
            new_mono_deprot.append(line[1])
        elif line[0] in di_set:
            assert line[1] in di_deprot_set
            assert line[0] not in mono_set
            # assert line[0] not in other_set
            new_di.append(line[0])
            assert line[1] not in mono_deprot_set
            new_di_deprot.append(line[1])
        elif line[0] in other_set:
            assert line[0] not in di_set
            assert line[0] not in mono_set
            new_other.append(line[0])
    fh.close()
    print('all_mono', len(new_mono), '.'.join(new_mono))
    print('all_di', len(new_di), '.'.join(new_di))
    # print('other', len(new_other), '.'.join(new_other))
    new_mono_noiso = remove_isotope_from_smiles_seq(new_mono)
    new_di_noiso = remove_isotope_from_smiles_seq(new_di)
    print('deprotected_mono', len(new_mono_deprot), '.'.join(new_mono_deprot))
    new_mono_noiso_deprot = remove_isotope_from_smiles_seq(new_mono_deprot)
    new_di_noiso_deprot = remove_isotope_from_smiles_seq(new_di_deprot)
    all_valids = dict()
    all_smi_gr = {'mono': new_mono, 'monoD': new_mono_deprot, 'monoNI': new_mono_noiso, 'monoDNI': new_mono_noiso_deprot,
                 'di': new_di, 'diD': new_di_deprot, 'diNI': new_di_noiso, 'diDNI': new_di_noiso_deprot}
    # for smigrname in all_smi_gr:
    #    for smi in all_smi_gr[smigrname]:
    #        if smi not in all_valids:
    #            all_valids[smi] = []
    #        all_valids[smi].append('smigrname')
    print('deprotected_di', len(new_di_deprot), '.'.join(new_di_deprot))
    print("NOISO_mono", '.'.join(new_mono_noiso))
    print("NOISO_di", '.'.join(new_di_noiso))
    print("NOISO_deprotected_mono", '.'.join(new_mono_noiso_deprot))
    print("NOISO_deprotected_di", '.'.join(new_di_noiso_deprot))
    if cutinfo:
        print_cutinfo_stats(cutinfo, all_smi_gr, heavy_mono_di, args)


def print_cutinfo_stats(cutinfo, allvalids, okheavy, args):
    new_cutinfo = dict()
    fh = open(f'{args.output}.fullcuts.csv', 'w')
    fhc = open(f'{args.output}.forcoverage.csv', 'w')
    fhcfull = open(f'{args.output}.forcoveragefull.csv', 'w')
    header = ('target', 'protectedtarget', 'numGens', 'monoDeprotected', 'diProtected', 'total#ofBuilings', 'blockAboveLimits')
    headerFull = ('target', 'protectedtarget', 'numGens', 'monoDeprotected', 'diDeprotected', 'total#ofBuilings', 'blockAboveLimits',
                  'monoProtected', 'diProtected')
    print(*header, sep=';', file=fh)
    print(*headerFull, sep=';', file=fhcfull)
    #print("XX", allvalids.keys())
    for target in cutinfo:
        tinfo = cutinfo[target]
        #print("TT", tinfo.keys()) # TT dict_keys(['mono', 'di', 'mono_deprotected', 'di_deprotected', 'other', 'numGens', 'protected'])
        valid_mono = [smi for smi in tinfo['mono_deprotected'] if smi in allvalids['monoD']]
        mono_protected = [smi for smi in tinfo['mono'] if smi in allvalids['mono']]
        valid_di = [smi for smi in tinfo['di'] if smi in allvalids['di']]
        monoset = set(tinfo['mono_deprotected'])
        diset = set(tinfo['di_deprotected'])
        heavy = len(okheavy.intersection(monoset.union(diset)))
        nblocks = len(valid_mono) + len(valid_di)
        datatoprint = (target, tinfo['protected'], tinfo['numGens'], '.'.join(valid_mono), '.'.join(valid_di), nblocks, heavy)
        valid_dide = [smi for smi in tinfo['di_deprotected'] if smi in allvalids['diD']]
        datafhc = (target, tinfo['protected'], tinfo['numGens'], '.'.join(valid_mono), '.'.join(valid_dide), nblocks, heavy)
        datafull = (target, tinfo['protected'], tinfo['numGens'], '.'.join(valid_mono), '.'.join(valid_dide), nblocks, heavy,
                    '.'.join(mono_protected), '.'.join(valid_di))
        new_cutinfo[target] = set(valid_mono).union(set(valid_di))
        print(*datatoprint, sep=';', file=fh)
        print(*datafhc, sep=';', file=fhc)
        print(*datafull, sep=';', file=fhcfull)
    fh.close()
    cutinfo = new_cutinfo
    zerocut = [target for target in cutinfo if len(cutinfo[target]) == 0]
    print(f"ZERO FRAGMENETS from {len(zerocut)} cmds :: {zerocut}")
    fh = open(f'{args.output}.nocuts.csv', 'w')
    print(len(zerocut), file=fh)
    print('.'.join(zerocut), file=fh)
    fh.close()
    lens = [len(cutinfo[target]) for target in cutinfo]
    print(f"# of compounds from target: MIN: {min(lens)} MAX: {max(lens)} AVG: {statistics.mean(lens)}")


def remove_isotope_from_smiles_seq(smiles_seq):
    results = []
    for smiles in smiles_seq:
        mol = Chem.MolFromSmiles(smiles)
        _ = [atm.SetIsotope(0) for atm in mol.GetAtoms()]
        smi = Chem.MolToSmiles(mol)
        results.append(smi)
    return results


if __name__ == "__main__":
    args = loader.parse_args()
    print("ARGS", args)
    rxn_list = loader.load_reactions(args)
    inco_info = loader.load_inco(args.incolist)
    drug_info = loader.parse_drug_list(args.druglist)
    killers = loader.load_killers(args.killers)
    if args.mode in {'drugsp3', 'drugnosp3', 'sp3sp2', 'sp3sp2cn', 'alldrugs'}:
        wrong_stereo_smiles = dict()
    if args.testnet:
        import tests
        tests.perform_testnet(make_retroanalysis, rxn_list, inco_info, drug_info, killers, args)
    elif args.smiles:
        targets = args.smiles.split('.')
        _, results, warnings, _, origins, cutinfo = make_retroanalysis(targets, rxn_list, inco_info, drug_info, killers, args)
        if results:
            print_results(results, warnings, origins, cutinfo, args)
        else:
            print("NO RES", results)
    elif args.makestats:
        numProds = []
        numGens = []
        targets, targets_to_drug = loader.load_targets(args, include_raw=True)
        nocuts = set()
        for target in targets:
            synAll, results, warnings, nSteps, _ = make_retroanalysis([target, ], rxn_list, inco_info, drug_info, killers, args)
            if not synAll:
                # nocuts.add(target)
                continue
            canonTarget = tuple(synAll.keys())[0]
            prods = synAll[canonTarget]
            if args.debug:
                print("CIECIE", target, '.'.join(prods), "nGens", nSteps[0], results)
            prods = len(results['mono']) + len(results['di'])
            print("NUMPROD:", prods, target)
            if prods == 0:
                nocuts.add(target)
            numProds.append(prods)
            numGens.append(nSteps[0])
        print("avg/min/max Prods", statistics.mean(numProds), min(numProds), max(numProds))
        print('avg/min/max nGens', statistics.mean(numGens), min(numGens), max(numGens))
        print(f"zero cuts for {len(nocuts)} cmds")
    else:
        targets, target_to_drug = loader.load_targets(args, include_raw=True)
        _, results, warnings, _, origins, cutinfo = make_retroanalysis(targets, rxn_list, inco_info, drug_info, killers,
                                                                       args, drug_si=target_to_drug)
        if results:
            print_results(results, warnings, origins, cutinfo, args)
