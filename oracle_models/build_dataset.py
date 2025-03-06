
from sklearn.preprocessing import MultiLabelBinarizer


from datasets import load_dataset

import pandas as pd


dataset_name = 'language-plus-molecules/LPM-24_train'

if dataset_name == 'language-plus-molecules/LPM-24_train': name = 'train'
if dataset_name == 'language-plus-molecules/LPM-24_eval-caption': name = 'valid'
if dataset_name == 'language-plus-molecules/LPM-24_eval-molgen': name = 'test'


# Load dataset from Hugging Face
dataset = load_dataset(dataset_name)  # Replace with actual dataset name

# Convert to pandas DataFrame (assuming you want the 'train' split)
df = dataset["train"].to_pandas()

## Display first few rows
#print(df.head())

mask = df['properties'].apply(lambda x: any(x == ['Mystery Molecule']))

df = df[~mask]

df.drop(columns=['caption', 'additional_data'], inplace=True, errors='ignore')

# Initialize MultiLabelBinarizer
mlb = MultiLabelBinarizer()

# Transform the 'properties' column into one-hot encoded format
one_hot = pd.DataFrame(mlb.fit_transform(df['properties']), columns=mlb.classes_)

one_hot.sum().sort_values(ascending=False).to_csv(f'counts_{name}.csv')

# Concatenate with original DataFrame (excluding 'properties' column if needed)
df = df.drop(columns=['properties']).join(one_hot)

# Display result
print(df)

df.to_csv(f'{name}.csv', index=False)




