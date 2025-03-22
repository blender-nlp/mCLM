
import json

from mCLM_tokenizer.tokenizer import get_blocks

from tqdm import tqdm
tqdm.pandas()


if __name__ == '__main__':

    import pandas as pd

    outpath = './'#'/shared/nas/data/m1/shared-resource/MoleculeLanguage/data/blockified/'
    file = 'train_input.txt'
    #file = 'test_input.txt'

    df = pd.read_csv(f"/shared/nas/data/m1/shared-resource/MoleculeLanguage/data/{file}", delimiter='\t', header=None)
    df.columns = ['name', 'description']

    kinases = pd.read_csv("/shared/nas/data/m1/shared-resource/MoleculeLanguage/data/kinase_inhibitor_list.txt", delimiter='\t')
    kinases.set_index('name', inplace=True)

    df['smiles'] = df['name'].apply(lambda x : kinases.loc[x]['smiles'])

    preprocessed = {}


    #df['blocks'] = df['name'].progress_apply(get_blocks)
    #df['rblocks'] = df['blocks'].progress_apply(reverse_rfrags)

    def app(x):
        if x in preprocessed: return preprocessed[x]
        else: 
            a = get_blocks(x)
            preprocessed[x] = a
            return a


    df[['blocks', 'rblocks']] = df['smiles'].progress_apply(lambda x: pd.Series(app(x)))

    print(df)

    df.to_csv(f"{outpath}{file}")

    if file == 'train_input.txt':

        out = []
        for s in preprocessed:
            for s2 in preprocessed[s]:
                for s3 in s2.split('^'):
                    out.append(s2.replace('2*', '*').replace('1*', '*'))


        counts = dict()
        for b in out:
            counts[b] = counts.get(b, 0) + 1
            
        for b in counts: counts[b] = int(counts[b]/2)

        counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        
        with open(outpath+'kinase_counts.txt', 'w') as convert_file: 
            convert_file.write(json.dumps(counts, indent=4))

        with open(outpath+'kinase_tokens.txt', 'w') as convert_file: 
            convert_file.write(json.dumps(preprocessed, indent=4))

