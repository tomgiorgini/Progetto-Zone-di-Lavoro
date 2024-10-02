import os
import sys
sys.path.append(os.path.abspath(os.curdir).split('respirazione')[0] + 'respirazione')
from config.definitions import ROOT_DIR
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearnex import patch_sklearn 
import copy
from sklearn.model_selection import train_test_split

patch_sklearn()

def create_dataset(datain,mode,split=True):
    X=[]
    y=[]
    data = copy.deepcopy(datain)
    if mode == 'full': # all features
       for key in data:
            data[key].loc[(data[key]['Load'] == 0) & (data[key]['label'] == 2), 'label'] = 3 
            label = data[key].pop('label')
            data[key].pop('t')
            for sample in np.asarray(data[key]):
                X.append(sample)
            for sample in label:
                y.append(sample)
    if mode == 'no-load': # all features minus 'Load'
        for key in data:
            data[key].loc[(data[key]['Load'] == 0) & (data[key]['label'] == 2), 'label'] = 3 
            label = data[key].pop('label')
            data[key].pop('t')
            data[key].pop('Load')
            for sample in np.asarray(data[key]):
                X.append(sample)
            for sample in label:
                y.append(sample)
    if mode == 'rf-hr': # only Frequenza di respirazione and Heartrate
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
            for sample in np.asarray(data[key]):
                X.append(sample)
            for sample in label:
                y.append(sample)

    X = np.asarray(X)
    y = np.asarray(y)
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
    params = {'C': [1,10,100,1000],  
              'gamma': [1, 0.1, 0.01, 0.001]
              } 
    scoring = ['accuracy','precision_macro','recall_macro','f1_macro']
    svc = svm.SVC()
    for m in modes:
        print("\nTUNING SVC WITH MODE: ",m)
        X_train, X_test, y_train, y_test = create_dataset(out_data,m,split=False)
        clf = GridSearchCV(svc,param_grid=params,scoring=scoring, refit='f1_macro')
        clf.fit(X_train,y_train)
        print("Migliori iperparametri:", clf.best_params_)
        print("Miglior score:", clf.best_score_)
        
#TUNING SVC WITH MODE:  full
#Migliori iperparametri: {'C': 1000, 'gamma': 0.1}
#Miglior score: 0.9513300956651507

#TUNING SVC WITH MODE:  no-load
#Migliori iperparametri: {'C': 1000, 'gamma': 1}
#Miglior score: 0.8992734995398957

#TUNING SVC WITH MODE:  rf-hr
#Migliori iperparametri: {'C': 1000, 'gamma': 1}
#Miglior score: 0.5259131056144924
