#!/usr/bin/env python
# coding: utf-8
import sys
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem
# import drawing_utlis
import rxn_mols_prep


def _load_rx(fname, args, with_rx_core=False):
    rxn_list = []
    with open(fname) as f:
        for line in f:
            if ';' in line:
                line = line.strip().split(';')
            elif '\t' in line:
                line = line.split('\t')
            else:
                print("LINE", line, "in", fname)
                raise NotImplementedError
            # 0:id; 1:rxsma; 2:.Br; 3:?; 4:[c][NH2]; 5:forwardrx   6: keyword, 7:prods to save
            info = {'rxid': line[0], 'rxname': line[1], 'rxsmarts': line[2], 'fix': line[3],
                    'inco': line[4], 'nonselect': line[5], 'forwardsmarts': line[6].strip(),
                    'retrorx': AllChem.ReactionFromSmarts(line[2]),
                    'fixedmols': [Chem.MolFromSmiles(smi) for smi in line[3].split('.') if smi],
                    }
            if with_rx_core:
                core_mol = info['retrorx'].GetReactantTemplate(0)
                info['core'] = Chem.MolFromSmarts(Chem.MolToSmarts(core_mol))
            rxn_list.append(info)
    if args.debug:
        print(f"{len(rxn_list)} reaction loaded from {fname}")
    return rxn_list


def _load_multiply_rx(fname, args, with_rx_core=True):
    rxn_list = []
    with open(fname) as f:
        for line in f:
            line = line.strip().split(';')
            # 0:id; 1:rxcore; 2:fixsbses; 3. rxsma
            info = {'rxid': line[0], 'rxcore': line[1],
                    'rxsmarts': line[3], 'rxn': AllChem.ReactionFromSmarts(line[3]),
                    'fix': line[2].split('.'),
                    'fixedmols': [Chem.MolFromSmiles(smi) for smi in line[2].split('.') if smi],
                    }
            if with_rx_core:
                info['core'] = Chem.MolFromSmarts(info['rxcore'])
            rxn_list.append(info)
    if args.debug:
        print(f"{len(rxn_list)} reaction loaded from {fname}")
    return rxn_list


def _load_retro_rx(fname, keywords, args, with_rx_core=False):
    rxn_list = []
    with open(fname) as f:
        for line in f:
            if '\t' in line:
                line = line.split('\t')
            elif ';' in line:
                line = line.strip().split(';')
            else:
                raise NotImplementedError
            # 0:id; 1:rxsma; 2:.Br; 3:?; 4:[c][NH2]; 5:forwardrx   6: keyword, 7:prods to save
            try:
                minsize = int(line[9].strip())
            except ValueError:
                if args.debug:
                    print("minsize not defined in file for ", len(line), line)
                minsize = 0
            except IndexError:
                minsize = 0
            info = {'rxid': line[0], 'rxname': line[1], 'rxsmarts': line[2], 'fix': line[3],
                    'inco': line[4], 'nonselect': line[5], 'forwardsmarts': line[6],
                    'retrorx': AllChem.ReactionFromSmarts(line[2]),
                    'fixedmols': [Chem.MolFromSmiles(smi) for smi in line[3].split('.') if smi],
                    'keyword': line[7].strip(), 'minsize': minsize}
            if line[8].strip():
                info['saveprod'] = [int(poz) - 1 for poz in line[8].split('.')]
            if with_rx_core:
                core_mol = info['retrorx'].GetReactantTemplate(0)
                info['core'] = Chem.MolFromSmarts(Chem.MolToSmarts(core_mol))
            if info['keyword'] not in keywords:
                if args.debug:
                    print("IGNORE RX", info)
                continue
            rxn_list.append(info)
    if args.debug:
        print(f"{len(rxn_list)} reaction loaded from {fname}")
    return rxn_list


def load_reactions(args):
    if args.retrokeywords in ('std', 'ext'):
        retrokeywords = args.retrokeywords
    elif args.retrokeywords == 'std_and_ext':
        retrokeywords = ('std', 'ext')
    elif args.retrokeywords == 'stdext':
        retrokeywords = ['std', ]
    else:
        raise NotImplementedError
    retro = _load_retro_rx(args.retrodb, retrokeywords, args)
    deprot = _load_rx(args.deprotdb, args)
    prot = _load_rx(args.protdb, args)
    initprot = _load_rx(args.initprotdb, args)
    finalprot = _load_rx(args.finalprotdb, args, with_rx_core=True)
    multiply = _load_multiply_rx(args.multiplydb, args, with_rx_core=True)
    ret_dict = {'retro': retro, 'protection': prot, 'deprotection': deprot,
                'init_protection': initprot, 'final_protection': finalprot,
                'multiply': multiply}
    if args.retrokeywords == 'stdext':
        ret_dict['retro2'] = retro = _load_retro_rx(args.retrodb, ['ext', ], args)
    return ret_dict


def load_killers(fn):
    ignored = set()
    if fn:
        fh = open(fn)
        for line in fh:
            for smi in line.split('.'):
                smi = smi.strip()
                if not smi:
                    continue
                canon = Chem.CanonSmiles(smi)
                if not canon:
                    raise NotImplementedError
                ignored.add(canon)
        fh.close()
    return ignored


def load_smiles_list(fn, toignore):
    smiles_list = []
    with open(fn) as f:
        for line in f:
            line = line.strip().split('.')
            for mol in line:
                # print(mol)
                try:
                    mol_sanit = Chem.MolToSmiles(Chem.MolFromSmiles(mol))
                except:
                    print(f"CANNOT PARSE SMILES {mol}", file=sys.stderr)
                    raise
                smiles_list.append(mol_sanit)
    return smiles_list


def load_targets(args, include_raw=False):
    retros_list = load_smiles_list(args.input, args.killers)
    # print('NUMBER OF TARGETS',len(retros_list))
    to_mark_smarts = ('[CX4][NH][CX4H2]', '[CX4,c][NH2]', '[c]@[NH]@[CX4]', '[nH]')
    to_mark = [Chem.MolFromSmarts(sma) for sma in to_mark_smarts]
    retros_final = []
    retro_canon = dict()
    for smiles in retros_list:
        smiles = Chem.CanonSmiles(smiles)
        mol = Chem.MolFromSmiles(smiles)
        found_patterns = set()
        for pattern in to_mark:
            for match in mol.GetSubstructMatches(pattern):
                found_patterns.update(set(match))
        for atm in mol.GetAtoms():
            if atm.GetSymbol() == 'N' and atm.GetIdx() in found_patterns:
                atm.SetIsotope(1)
        canonSmiles = Chem.MolToSmiles(mol)
        retros_final.append(canonSmiles)
        retro_canon[canonSmiles] = smiles
    retros_list = rxn_mols_prep.sanitize(retros_final)
    print("targets:", len(retros_list), '.'.join(retros_list))
    if include_raw:
        return retros_list, retro_canon
    return retros_list


def getNames(retros_list, drug_info, drug_si):
    names = []
    for smi in retros_list:
        if smi in drug_info:
            names.append(drug_info[smi])
        elif drug_si and smi in drug_info:
            oryg = drug_si[smi]
            names.append(drug_info[oryg])
        else:
            pass
    return names


def load_inco(fn):
    inco_info = dict()
    for line in open(fn):
        name, sma_list = line.split('\t')
        sma_list = sma_list.strip().split('.')
        mol_list = [Chem.MolFromSmarts(sma) for sma in sma_list]
        inco_info[name] = {'smarts': sma_list, 'mols': mol_list}
    return inco_info


def parse_drug_list(fname):
    if not fname:
        return dict()
    fh = open(fname)
    smidict = dict()
    for lid, line in enumerate(fh):
        if lid == 0:
            continue
        name, _, smiles = line.split('\t')
        canon = Chem.CanonSmiles(smiles)
        smidict[canon] = name.strip()
    fh.close()
    return smidict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='kinase', choices=['kinase', 'alldrugs', 'with_suzuki_sp3_sp2'],
                        help='kinase - db and blacklists for kinase inhibitor cuts; alldrugs - general mode for drug cuts'
                        'with_suzuki_sp3_sp2 - database with aryl/vinyl-sp3 Suzuki couplings, no stereo')
    parser.add_argument('--retrodb', type=str, default='', help='allow manual selection of reaction database. '
                        '*** WARNING: *** Correct results are obtained only when all databases (--retrodb --deprotdb --protdb --multiplydb --initprotdb'
                        ' --finalprotdb and --incolist) match to each other. For common usecase use --mode instead.')
    parser.add_argument('--retrokeywords', type=str, default='stdext', choices=['ext', 'std', 'stdext', 'std_and_ext'],
                        help='stdext = perform std first and then ext; std_and_ext = combine both and perform calculations')
    parser.add_argument('--deprotdb', type=str, default='deprotdb_modulizer.csv')
    parser.add_argument('--protdb', type=str, default='protdb_modulizer.csv')
    parser.add_argument('--initprotdb', type=str, default='init_protection_kinase.csv', help='init_protection_without_bioisosteric.csv - without bioisosteric replacements')
    parser.add_argument('--finalprotdb', type=str, default='final_protection_kinase.csv', help='final_protection_without_bioisosteric.csv - without bioisosteric replacements')
    parser.add_argument('--incolist', type=str, default='', help='file with incompatibility')
    parser.add_argument('--multiplydb', type=str, default='multiply_rx.csv')
    parser.add_argument('--output', type=str, default='results_from_modulizer', help='prefix for output files (suffix .csv will be added to all files)')
    parser.add_argument('--testnet', action='store_true')
    parser.add_argument('--debug', action='store_true', help='print tons of debuging informations to stdout')
    parser.add_argument('--makestats', action='store_true')
    parser.add_argument('--smiles', type=str, default='')
    parser.add_argument('--killers', type=str, default='',
                        help='file with smiles which are problematic and will be ignored if found in input data')
    parser.add_argument('--numprocs', type=int, default=8)
    parser.add_argument('--timeout', type=int, default=0, help='give up after given time of unsuccessful waiting for results. Time given in seconds')
    parser.add_argument('--druglist', type=str, default='kinases_drugs.tsv', help='translate smiles strings into drug names')
    parser.add_argument('--maxC', type=int, default=19, help='max allowed number of C atoms in valid building blocks, protecting groups not included in limit')
    parser.add_argument('--progress', action='store_true', help='print progress information to stderr')
    parser.add_argument('--replaceMode', type=str, choices=['methylAll', 'noReplace'], default='methylAll',
                        help='methylAll - replace aromatic Br, Cl and I from target with methyl group; noReplace - do not perform replacement of aromatic halides')
    parser.add_argument('--swap', type=str, default='all', choices=['all', 'none'],
                        help='define whether swap reaction should be used: all - use all swap reaction; none -dont use at all')
    parser.add_argument('--input', type=str, default='Kinases_merged_OrangeBook', help='file with input smiles')
    parser.add_argument('--diazodef', type=str, choices=['general', 'r6'], default='general', help='definition of not allowed nc(B)n fragment')
    parser.add_argument('--checkParent', type=str, choices=['y', 'n'], default='n', help='include bigger block (i.e. such with can be cut) when'
                        ' one or more of formed block(s) is invalid')
    args = parser.parse_args()
    if args.retrodb == '':
        if args.mode == 'kinase':
            args.retrodb = 'retrodb_kinase_modularisation.csv'
            args.incolist = 'incolist_kinase_modularisation.csv'
        elif args.mode == 'with_suzuki_sp3_sp2':
            args.retrodb = 'retrodb_with_sp2_sp3.csv'
            args.incolist = 'incolist_with_sp2_sp3.csv'
    if args.incolist == '':
        if args.mode == 'kinase':
            args.incolist = 'incolist_kinase_modularisation.csv'
        elif args.mode == 'with_suzuki_sp3_sp2':
            args.incolist = 'incolist_with_sp2_sp3.csv'
    # developer options, hardcoded here for convenience
    args.heavyMode = 'maxC'  # definion of heavy block in output file
    args.warning = 'full'
    args.includeHardcodedCuts = ''
    args.maxgen = 50  # limit of target cutting generations
    return args
