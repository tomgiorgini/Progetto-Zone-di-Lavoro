import os
import sys
sys.path.append(os.path.abspath(os.curdir).split('respirazione')[0] + 'respirazione')
from config.definitions import ROOT_DIR
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from pickle import dump
import copy
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from sklearnex import patch_sklearn 

patch_sklearn()

def preprocessing(datain):
    data = copy.deepcopy(datain)
    for key in data:
        data[key].pop('t')
        data[key]= data[key].drop(data[key][data[key]['Load'] == 0].index)
        data[key]= data[key].drop(data[key][data[key]['HR'] == 0].index)
        data[key]= data[key].drop(data[key][data[key]['RF'] == 0].index)
        data[key]= data[key].drop(data[key][data[key]['VO2'] == 0].index)
        data[key]= data[key].drop(data[key][data[key]['VCO2'] == 0].index)
        data[key]= data[key].drop(data[key][data[key]['VE/VO2'] == 0].index)
        data[key]= data[key].drop(data[key][data[key]['VE/VCO2'] == 0].index)
        data[key]= data[key].drop(data[key][data[key]['VO2/HR'] == 0].index)
        data[key]= data[key].drop(data[key][data[key]['Q'] == 0].index)
        data[key].pop('Load')
        for col in data[key]:
            if col != 'label':
                data[key][col] = data[key][col].rolling(window=7,min_periods=1).mean()
    return data

if __name__ == "__main__":
    filename = 'sequence_dataset.pkl'
    with open(Path(ROOT_DIR) / f'progetto/dati_finali/training/{filename}','rb') as f:
        out_data = pickle.load(f)
    features = ['RF','HR','VO2','VCO2','VE/VO2','VE/VCO2','VO2/HR','Q']
    modes = ['RF-HR-VO2-VCO2-VE/VO2-VE/VCO2-Q-VO2/HR','VO2-VCO2','VE/VO2-VE/VCO2','RF-HR','Q-HR','Q-RF','Q-VO2','Q-VCO2','Q','HR']
    
    data = preprocessing(out_data)
    
    df = []
    # dataset
    for m in modes:
        temp = m.split('-')
        to_delete = set(temp) ^ set(features)
        X=[]
        y=[]
        for key in data:
            temp = copy.deepcopy(data[key])
            label = temp.pop('label')
            for e in to_delete:
                temp.pop(e)
            for sample in np.asarray(temp):
                X.append(sample)
            for sample in label:
                y.append(sample)
        X = np.asarray(X)
        y = np.asarray(y)
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]    
        df.append([m,X,y])
    df[0][0] = 'FULL'
    filename = 'dataset.pkl'
    with open(Path(ROOT_DIR) / f'progetto/finale/{filename}', 'wb') as f:
        dump(df, f)