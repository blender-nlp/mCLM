



import pandas as pd

import random

import json

import re

import unicodedata

from tqdm import tqdm
tqdm.pandas()


# Load CSV files
print('Loading files')

df1 = pd.read_csv("train_input.txt", index_col=0)
df2 = pd.read_csv("valid_input.txt", index_col=0)
df3 = pd.read_csv("test_input.txt", index_col=0)

mapping = df1.set_index('name')['blocks'].to_dict()
rmapping = df1.set_index('name')['rblocks'].to_dict()

#df1 = df1.head(10000)

del mapping['sirolimus']
del mapping['everolimus']
del rmapping['sirolimus']
del rmapping['everolimus']

# Print dictionary
#print(mapping)


def replace_mask(text, name):
    return text.replace('[MASK]', name)

def normalize_text(text):
    return unicodedata.normalize('NFKC', text).replace('gefıtinib', 'gefitinib')  # Converts similar characters to a standard for


def replace_with_mapping(text):

    if random.random() > 0.5: m = mapping
    else: m = rmapping

    # Escape keys to avoid regex issues and join them with '|'
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, m.keys())) + r')\b', re.IGNORECASE)

    text = normalize_text(text)

    # Replace matched keys with corresponding values
    try:
        return pattern.sub(lambda match: '[MOL] ' + m[match.group(0).lower()] + ' [/MOL]', text)
    except:
        print(text)
        zz

print('Running regex replace')

df1['description'] = df1.progress_apply(lambda row: replace_mask(row['description'], row['name']), axis=1)
df2['description'] = df2.progress_apply(lambda row: replace_mask(row['description'], row['name']), axis=1)
df3['description'] = df3.progress_apply(lambda row: replace_mask(row['description'], row['name']), axis=1)

df1['description'] = df1['description'].progress_apply(replace_with_mapping)
df2['description'] = df2['description'].progress_apply(replace_with_mapping)
df3['description'] = df3['description'].progress_apply(replace_with_mapping)

if False:
    print('Deduplication')
    # Assume df1, df2, and df3 are your DataFrames, and 'ColumnA' is the column to deduplicate
    column_name = 'description'

    # Step 1: Concatenate the column values from all three DataFrames
    combined = pd.concat([df1[column_name], df2[column_name], df3[column_name]], ignore_index=True)

    # Step 2: Remove duplicates
    unique_values = combined.drop_duplicates()

    # Step 3: Reassign the cleaned column back to the DataFrames
    df1[column_name] = df1[column_name].progress_apply(lambda x: x if x in unique_values.values else None)
    df2[column_name] = df2[column_name].progress_apply(lambda x: x if x in unique_values.values else None)
    df3[column_name] = df3[column_name].progress_apply(lambda x: x if x in unique_values.values else None)


df1 = df1.drop(['blocks', 'rblocks', 'name'], axis=1)
df2 = df2.drop(['blocks', 'rblocks', 'name'], axis=1)
df3 = df3.drop(['blocks', 'rblocks', 'name'], axis=1)

# Save DataFrames to CSV files
df1.to_csv("kinase_train.csv")
df2.to_csv("kinase_valid.csv")
df3.to_csv("kinase_test.csv")

with open('mapping.jsonl', 'w') as f:
    for key, value in mapping.items():
        json.dump({key: value}, f)
        f.write('\n')

with open('rmapping.jsonl', 'w') as f:
    for key, value in rmapping.items():
        json.dump({key: value}, f)
        f.write('\n')






