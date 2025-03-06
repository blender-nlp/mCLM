
from sklearn.preprocessing import MultiLabelBinarizer


from datasets import load_dataset

import pandas as pd


THRESH = 0

dfs = []

for dataset_name in ['language-plus-molecules/LPM-24_train', 'language-plus-molecules/LPM-24_eval-caption', 'language-plus-molecules/LPM-24_eval-molgen']:


    # Load dataset from Hugging Face
    dataset = load_dataset(dataset_name)  # Replace with actual dataset name

    # Convert to pandas DataFrame (assuming you want the 'train' split)
    df = dataset["train"].to_pandas()

    ## Display first few rows
    #print(df.head())

    mask = df['properties'].apply(lambda x: any(x == ['Mystery Molecule']))

    df = df[~mask]

    df.drop(columns=['caption', 'additional_data'], inplace=True, errors='ignore')

    dfs.append(df)

df = pd.concat(dfs)

if False:
    for row in df.iterrows():
        r = row[1]
        #print(r)
        if 'choleretic' in r['properties']: 
            print(r['molecule'], r['properties'])
        #zz
    zz



# Initialize MultiLabelBinarizer
mlb = MultiLabelBinarizer()

# Transform the 'properties' column into one-hot encoded format
one_hot = pd.DataFrame(mlb.fit_transform(df['properties']), columns=mlb.classes_)

counts = one_hot.sum().sort_values(ascending=False)

counts.to_csv(f'counts.csv')


count_mask = counts >= THRESH

# Concatenate with original DataFrame (excluding 'properties' column if needed)
df = df.drop(columns=['properties']).join(one_hot)

false_indices = count_mask.index[~count_mask]

df.drop(columns = false_indices, inplace=True)

# Display result
print(df)

df.to_csv(f'data.csv', index=False)




