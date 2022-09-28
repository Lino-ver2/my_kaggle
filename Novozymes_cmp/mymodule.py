import glob
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


# ############## featuring ################
def base_feature(df, name, save=False):
    # add protein length
    df['sequence_len'] = list(map(len, df['protein_sequence']))
    # count amino_acid appearances
    string = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for amino in string:
        df[amino] = 0
    idx = 0
    for trg in df['protein_sequence']:
        for amino in string:
            df[amino].loc[idx] = trg.count(amino)
        idx += 1

    if save:
        with open(f'dataset/featured/{name}.pkl', 'wb') as f:
            pickle.dump(df, f)
    return df


def except_outlier(df, target_name):
    q = df[target_name].describe()
    outlier = q.loc['75%'] + (q.loc['75%'] - q.loc['25%']) * 1.5
    outlier
    df = df[df[target_name] < outlier].reset_index(drop=True)
    return df


def one_hot_protein(df, name, save=False):
    """
    df: df['protein_sequence']
    name: string (pickle_name)
    """
    string = "ACDEFGHIKLMNPQRSTVWY"
    kinds = np.array([i for i in string]).reshape(-1, 1)

    enc = OneHotEncoder(categories="auto", sparse=False, dtype=np.float32)
    encoded = enc.fit_transform(kinds)

    amino_dic = {i: j for i, j in zip(kinds.reshape(-1), encoded)}
    featured = []
    for protein in df['protein_sequence']:
        oh_amino = []
        for amino in protein:
            oh_amino.append(amino_dic[amino])
        featured.append(np.array(oh_amino))
    df = np.array(featured)

    if save:
        with open(f'dataset/featured/{name}.pkl', 'wb') as f:
            pickle.dump(df, f)
    return df


# ############## separate data ###########
def pre_train_test(df, train, drops):
    idx = df['seq_id']
    x = df.drop(['seq_id', 'data_source'], axis=1)
    if train:
        x = x.drop(['tm'] + drops, axis=1)
        y = df['tm']
    else:
        x = x.drop(drops, axis=1)
        y = None
    return idx, x, y


# ############### evaluate ###################
def spearman_and_mse(y_truth, y_pred):
    y_truth = np.array(y_truth)
    y_pred = np.array(y_pred)
    N = len(y_truth)
    mse = sum((y_truth - y_pred)**2) / N
    spearman = 1 - (6*mse / (N**2 - 1))
    return spearman, mse


# ################ submit #####################
def load_model(model_path):
    files = glob.glob(model_path)

    models = []
    for file in files:
        with open(file, 'rb') as f:
            model = pickle.load(f)
        models.append(model)
    return models


def ensemble(models, x):
    pred_list = []
    for model in models:
        y_pred = model.predict(x)
        pred_list.append(y_pred)
    tm = pd.Series(np.round(np.mean(pred_list, axis=0), 1), name='tm')
    return tm


def pre_submit(idx, tm, submit_path):
    submit = pd.concat((idx, tm), axis=1)
    submit = submit.sort_values('tm', ascending=True)
    submit['tm'] = range(len(submit))
    submit.to_csv(submit_path, index=False)
