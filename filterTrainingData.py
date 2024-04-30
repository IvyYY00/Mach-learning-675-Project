import numpy as np


def check_if_idealized(row):

    mask = [row['seizure_vote'] > 0,row['lpd_vote'] > 0,row['gpd_vote'] > 0,row['lrda_vote'] > 0,row['grda_vote'] > 0,row['other_vote'] > 0]
    return sum(mask)==1


def check_unnecessary_row(row1, row2):
    '''
    row1: some row in train.csv
    row2: the next row of row1
    return True if row2 needs to be removed from train.csv
    '''
    if row1['eeg_id']==row2['eeg_id'] and row1['eeg_label_offset_seconds']+1==row2['eeg_label_offset_seconds']:        
        if row1['expert_consensus']==row2['expert_consensus']:
            return True
    return False

def get_true_type_of_patient(row):
    '''
    input: one row in patient_type dataframe.
    return one of "GPD	GRDA	LPD	LRDA	Other	Seizure" if one takes percentage more than 75%.
    otherwise return None.
    '''
    row = row.drop('patient_id')
    total_cnt = row['GPD']+row['GRDA']+row['LPD']+row['LRDA']+row['Other']+row['Seizure']
    row = row/total_cnt
    greater_than_075 = row > 0.75 
    lst= greater_than_075[greater_than_075].index.tolist()
    
    if len(lst) >0:
        return lst[0]
    else:
        return None

import pandas as pd

df = pd.read_csv('train.csv')

# for each patient, get their true type
patient_dist = df.groupby('patient_id')['expert_consensus'].value_counts().unstack(fill_value=0)
patient_dist.reset_index(inplace=True)
patient_dist['patient_true_type'] = patient_dist.apply(get_true_type_of_patient, axis=1)
# print(patient_dist.columns)

patient_type_map={}
for i in range(len(patient_dist) - 1):
    row = patient_dist.iloc[i]
    # some patient may end up not being assigned a type
    # delete those patients too
    if row['patient_true_type'] is not None:
        patient_type_map[row['patient_id']]=row['patient_true_type']


idx_wrong_patient_lbl_or_undicided_patient=[]
for i in range(len(df) - 1):
    row = df.iloc[i]
    if row['patient_id'] not in patient_type_map or patient_type_map[row['patient_id']] != row['expert_consensus']:
        idx_wrong_patient_lbl_or_undicided_patient.append(i)

df = df.drop(idx_wrong_patient_lbl_or_undicided_patient).reset_index(drop=True)

columns_to_sum = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
# there must be at least 5 experts to vote
df = df[df[columns_to_sum].sum(axis=1) >= 5]
#only train on idealized votes
df = df[df.apply(check_if_idealized, axis=1)]

indices_to_remove = []
# Iterate over DataFrame rows in pairs
for i in range(len(df) - 1):
    row1 = df.iloc[i]
    row2 = df.iloc[i+1]
    if check_unnecessary_row(row1, row2):
        indices_to_remove.append(i+1)
    if i%5000==0:
        print("processed 5000 rows")
df_filtered = df.drop(indices_to_remove).reset_index(drop=True)

df_filtered.to_csv("filtered_train.csv")
