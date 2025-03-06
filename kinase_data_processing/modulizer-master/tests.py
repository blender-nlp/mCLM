import urllib, os
from rdkit import Chem

def downloadFile(addr):
    data = urllib.request.urlopen(addr).read().decode()
    suffix = 'tsv'
    ver = 0
    basename = 'downloaded_data_test_for_forward'
    while True:
        if not ver:   # 0 is false
            name = f'{basename}.{suffix}'
        else:
            name = f'{basename}_{ver}.{suffix}'
        if not os.path.exists(name):
            break
        ver += 1
    fw = open(name, 'w')
    fw.write(data)
    fw.close()
    return name


def perform_testnet(calc_function, reactions, inco_info, debug, warning):
    addr = 'https://docs.google.com/spreadsheets/d/1fvJngeY4T8FC9sfoEaN7QJbTFFZdSJS0iMjFh09hGT0/export?gid=0&format=tsv'
    fn = downloadFile(addr)
    for line in open(fn).readlines()[1:]:
        line = line.strip()
        if not line:
            continue
        try:
            target, mono, di = line.split('\t')
        except:
            print("INCORRECT LINE", line)
            continue
        mono = [Chem.CanonSmiles(smi) for smi in mono.split('.')]
        di = [Chem.CanonSmiles(smi) for smi in di.split('.')]
        _, res, _, _ = calc_function([target, ], reactions, inco_info, debug, warning)
        print("===")
        print("Full RESULTS::", res)
        # found = set(res.keys())
        not_found_mono = [m for m in mono if not(m in res['mono'] or m in res['mono_deprotected'])]
        not_found_di = [m for m in di if not(m in res['di'] or m in res['di_deprotected'])]
        extra_mono = [smi for smi in res['mono_deprotected'] if smi not in mono]
        extra_di = [smi for smi in res['di_deprotected'] if smi not in di]
        print(target, "==> NotFound:", not_found_mono, not_found_di, "EXTRA:", '.'.join(extra_mono), '.'.join(extra_di), "OTHER", '.'.join(res['other']))
        print("MONO: EXP::", len(mono), '.'.join(mono), "FOUND::", len(res['mono_deprotected']), '.'.join(res['mono_deprotected']))
        print("DI: EXP::", len(di), '.'.join(di), "FOUND::", len(res['di_deprotected']), '.'.join(res['di_deprotected']))
        print("===")
