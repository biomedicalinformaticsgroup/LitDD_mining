import os
import pandas as pd
from datasets import Dataset, Features, Value, ClassLabel
from sklearn.model_selection import train_test_split

df_final = pd.read_csv('annotated_tiab.csv')

# group-level split to avoid TIAB leakage
tiab_group = df_final.groupby('tiab', as_index=False)['label'].max()
tiab_group.rename(columns={'label': 'has_pos'}, inplace=True)

# Stratified split on has_pos at the TIAB level
train_tiab, test_tiab = train_test_split(
    tiab_group,
    test_size=0.20,
    random_state=42,
    stratify=tiab_group['has_pos']
)

train_tiabs = set(train_tiab['tiab'])
test_tiabs = set(test_tiab['tiab'])

# row-level filtering using the group split
df_train = df_final[df_final['tiab'].isin(train_tiabs)].copy()
df_test  = df_final[df_final['tiab'].isin(test_tiabs)].copy()

# sanity check: no TIAB overlap
assert train_tiabs.isdisjoint(test_tiabs), "TIAB leakage detected between train and test!"

df_train = df_train[["tiab", "g2p_lgmde", "label"]]
df_test  = df_test[["tiab", "g2p_lgmde", "label"]]

# create HF Datasets with consistent label schema 
features = Features({
    "tiab": Value("string"),
    "g2p_lgmde": Value("string"),
    "label": ClassLabel(num_classes=2)  # 0/1
})

ds_train = Dataset.from_pandas(df_train, preserve_index=False).cast(features)
ds_test  = Dataset.from_pandas(df_test,  preserve_index=False).cast(features)

# same train set for BERT and CrossEncoder
ds_bert  = ds_train
ds_cross = ds_train

ds_bert.save_to_disk('ds_bert_train')
ds_cross.save_to_disk('ds_cross_train')
ds_test.save_to_disk('ds_test')

print(f'ds bert train saved; size {len(ds_bert)} (TIABs: {df_train.tiab.nunique()})')
print(f'ds cross train saved; size {len(ds_cross)} (TIABs: {df_train.tiab.nunique()})')
print(f'ds test saved; size {len(ds_test)} (TIABs: {df_test.tiab.nunique()})')

