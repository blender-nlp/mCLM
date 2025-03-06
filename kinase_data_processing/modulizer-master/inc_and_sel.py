from rdkit import Chem
from rdkit.Chem import AllChem
import rxn_mols_prep


def identify_incomatibilities(mol, inc_list, core):
    for inc_group in inc_list:
        mol_wo_core = Chem.ReplaceCore(mol, core, labelByIndex=True)
        inc_group = Chem.MolFromSmarts(inc_group)
        if mol_wo_core.HasSubstructMatch(inc_group):
            return True
    return False


def identify_double_halogene(mol):
    substruct = ['c[Cl,I]', ]
    for s in substruct:
        check = Chem.MolFromSmiles(mol).HasSubstructMatch(Chem.MolFromSmarts(s))
        if check:
            return True
    return False


def GetRingSystems(mol, includeSpiro=False):
    ri = mol.GetRingInfo()
    systems = []
    for ring in ri.AtomRings():
        ringAts = set(ring)
        nSystems = []
        for system in systems:
            nInCommon = len(ringAts.intersection(system))
            if nInCommon and (includeSpiro or nInCommon > 1):
                ringAts = ringAts.union(system)
            else:
                nSystems.append(system)
        nSystems.append(ringAts)
        systems = nSystems

    # detects TIDA and substract's it from the rings count
    tida = '[#6][B]2OC(=O)C([CH3])([CH3])[N]([CH3])C([CH3])([CH3])C(=O)O2'
    cbz_tida = 'CN1C(C)(C)C(=O)OB(OC(=O)C1(C)C)c1ccc(COC=O)cc1'
    res = len(systems)
    if mol.HasSubstructMatch(Chem.MolFromSmarts(tida)) and not mol.HasSubstructMatch(Chem.MolFromSmarts(cbz_tida)):
        tida_motif = Chem.MolFromSmarts(tida)
        tida_count = len(mol.GetSubstructMatches(tida_motif))
        print('substr', tida_count)
        res = len(systems) - tida_count
    elif mol.HasSubstructMatch(Chem.MolFromSmarts(cbz_tida)):
        cbz_tida_motif = Chem.MolFromSmarts(cbz_tida)
        cbz_tida_count = len(mol.GetSubstructMatches(cbz_tida_motif))
        res = len(systems) - (2 * cbz_tida_count)
        print('substr cbz_tida', cbz_tida_count, res, Chem.MolToSmiles(mol))
    return res


def count_Products_From_Rxn(rxn):
    left = rxn.split('>>')
    expected_products_no = len(left[0].split('.'))
    return expected_products_no


def is_selective(rxn, sel_groups, rxn_rev, retron_mol, small_retrons, debug=False):
    subs_sel = [retron_mol, ]
    for c in small_retrons:
        if not c:
            continue
        c_mol = Chem.MolFromSmiles(c)
        subs_sel.append(c_mol)
    if debug:
        for mol in subs_sel:
            print('Subs', Chem.MolToSmiles(mol))
    reverse = AllChem.ReactionFromSmarts(rxn_rev)
    left_motifs = rxn_rev.split('>>')[0].split('.')
    if debug:
        print('FORWARD RXN', rxn_rev)
        print('RETRO RXN', rxn)
    retro_rxn = AllChem.ReactionFromSmarts(rxn)
    substrates_tuple = tuple(subs_sel)

    ps = retro_rxn.RunReactants(substrates_tuple)
    unique = set()
    for p in ps:
        tmp = set()
        for mol in p:
            mol_MolFile = Chem.MolToSmiles(mol)
            tmp.add(mol_MolFile)
        unique.add(tuple(sorted(tmp)))
    if debug:
        print('UNIQUE PAIRS OF SUSBTRATES', unique)

    ordered_substrates = set()
    ordered_smileses_of_substrates = set()
    for pair_subs in unique:
        tmp3 = []
        tmp4 = []
        for motif in left_motifs:
            motif_mol = Chem.MolFromSmarts(motif)
            for m in pair_subs:
                m_sanitized = Chem.MolFromSmiles(rxn_mols_prep.sanitize_mol(m))
                if m_sanitized.HasSubstructMatch(motif_mol):
                    tmp3.append(m_sanitized)
                    tmp4.append(m)
        if len(pair_subs) == len(tmp3):
            ordered_substrates.add(tuple(tmp3))
            ordered_smileses_of_substrates.add(tuple(tmp4))
        else:
            if debug:
                print('not selective reaction:', tmp4)
    if debug:
        print('TMP4-----------------', ordered_smileses_of_substrates)
    # ordered_substrates_tuple = tuple(ordered_substrates)
    # fwd_products = reverse.RunReactants(ordered_substrates_tuple)

    fwd_products_set = set()
    products_no = count_Products_From_Rxn(rxn)
    for cmpds_pair in ordered_substrates:
        fwd_ps = reverse.RunReactants(cmpds_pair)
        for pair in fwd_ps:
            for mol in pair:
                p_smiles = Chem.MolToSmiles(mol)
                fwd_products_set.add(p_smiles)
                if len(fwd_products_set) > products_no:
                    if debug:
                        print('non-selective', len(fwd_products_set))
                    return False
    if debug:
        if len(fwd_products_set) == products_no:
            print('selective')
        else:
            print('not selective: no products')
    return True
