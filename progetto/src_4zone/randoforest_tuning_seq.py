import os
import sys
sys.path.append(os.path.abspath(os.curdir).split('respirazione')[0] + 'respirazione')
from config.definitions import ROOT_DIR
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearnex import patch_sklearn 
import copy
from collections import Counter
from sklearn.model_selection import train_test_split

patch_sklearn()


def sequence_create(datain,mode,seq_len,n_samples,split = True):
    X=[]
    y=[]
    data = copy.deepcopy(datain)
    l = len(data.keys())
    samplesxset = round(n_samples /l) # calcolo il numero di sequenze da campionare per ogni set di sample
    if mode == 'full': # all features
        for key in data:
            data[key].loc[(data[key]['Load'] == 0) & (data[key]['label'] == 2), 'label'] = 3 
            label = data[key].pop('label')
            data[key].pop('t')
            max_idx = len(data[key]) - seq_len # massimo indice da cui far partire la sequenza
            for i in range(samplesxset):
                idx = np.random.randint(max_idx,size = 1)[0]
                x_seq = np.asarray(data[key])[idx:idx+seq_len]
                temp = []
                for e in x_seq:
                    for a in e:
                        temp.append(a) # temp è un vettore uni-dimensionale che contiene tutta la sequenza
                y_seq = Counter(np.asarray(label)[idx:idx+seq_len]).most_common(1)[0][0] # prendo come label la più comune 
                X.append(temp)
                y.append(y_seq)
    if mode == 'no-load': # all features minus 'Load'
        for key in data:
            data[key].loc[(data[key]['Load'] == 0) & (data[key]['label'] == 2), 'label'] = 3 
            label = data[key].pop('label')
            data[key].pop('t')
            data[key].pop('Load')
            max_idx = len(data[key]) - seq_len
            for i in range(samplesxset):
                idx = np.random.randint(max_idx,size = 1)[0]
                x_seq = np.asarray(data[key])[idx:idx+seq_len]
                temp = []
                for e in x_seq:
                    for a in e:
                        temp.append(a)
                y_seq = Counter(np.asarray(label)[idx:idx+seq_len]).most_common(1)[0][0]
                X.append(temp)
                y.append(y_seq)
    if mode == 'rf-hr':  # only Frequenza di respirazione and Heartrate
        for key in data:
            data[key].loc[(data[key]['Load'] == 0) & (data[key]['label'] == 2), 'label'] = 3 
            label = data[key].pop('label')
            data[key].pop('t')
            data[key].pop('Load')
            data[key].pop('VO2')
            data[key].pop('VCO2')
            data[key].pop('VE/VO2')
            data[key].pop('VE/VCO2')
            data[key].pop('VO2/HR')
            max_idx = len(data[key]) - seq_len
            for i in range(samplesxset):
                idx = np.random.randint(max_idx,size = 1)[0]
                x_seq = np.asarray(data[key])[idx:idx+seq_len]
                temp = []
                for e in x_seq:
                    for a in e:
                        temp.append(a)
                y_seq = Counter(np.asarray(label)[idx:idx+seq_len]).most_common(1)[0][0]
                X.append(temp)
                y.append(y_seq)
    X = np.asarray(X)
    y = np.asarray(y)
    
    df = pd.DataFrame(X)
    df['label'] = y         # rimuovo le sequenza duplicate che si possono essere create
    df = df.drop_duplicates()
    
    y = np.asarray(df.pop('label'))
    X = np.asarray(df)
    if split: # split per il training e testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        return X_train,X_test,y_train,y_test
    else: # per il tuning
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)
        return X_train,X_test,y_train,y_test
    
if __name__ == "__main__":
    filename = 'sequence_dataset.pkl'
    with open(Path(ROOT_DIR) / f'progetto/dati_originali/{filename}','rb') as f:
        out_data = pickle.load(f)
    modes = ['full','no-load','rf-hr']
    params = {
        'max_features': ['sqrt', 'log2'], 
        'max_depth': [None], 
        'max_leaf_nodes': [None],
        'min_samples_split': [2,3,4,5,6]
    }
    scoring = ['accuracy','precision_macro','recall_macro','f1_macro']
    rf = RandomForestClassifier()
    seq_len = [2,4,8,16,32]
    filename = 'rf_tuning.txt'
    with open(Path(ROOT_DIR) / f'progetto/src_4zone/output/{filename}',"w") as f:
        print('',file=f)
    for l in seq_len:
        print("\nTUNING RANDOM FOREST WITH SEQ_LEN = ",l)
        with open(Path(ROOT_DIR) / f'progetto/src_4zone/output/{filename}',"a") as f:
                    print("\nTUNING RANDOM FOREST WITH SEQ_LEN = ",l,file=f)
        for m in modes:
            print("\nWITH MODE: ",m)
            with open(Path(ROOT_DIR) / f'progetto/src_4zone/output/{filename}',"a") as f:
                    print("\nWITH MODE",m,file=f)
            X_train, X_test, y_train, y_test = sequence_create(out_data,m,l,10000,split=False)
            clf = GridSearchCV(rf,param_grid=params,scoring=scoring, refit='f1_macro')
            clf.fit(X_train,y_train)
            print("Migliori iperparametri:", clf.best_params_)
            print("Miglior score:", clf.best_score_)
            with open(Path(ROOT_DIR) / f'progetto/src_4zone/output/{filename}',"a") as f:
                    print("Migliori iperparametri:", clf.best_params_,file=f)
                    print("Miglior score:", clf.best_score_,file=f)