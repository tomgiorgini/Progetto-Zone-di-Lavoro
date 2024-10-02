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
from collections import Counter
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
    seq_len = [2,4,8,16,32]
    data = preprocessing(out_data)
    
    df = []
    for l in seq_len:
        for m in modes:
            temp = m.split('-')
            to_delete = set(temp) ^ set(features)
            X=[]
            y=[]
            for key in data:
                max_idx = len(data[key]) - l
                temp = copy.deepcopy(data[key])
                label = temp.pop('label')
                for e in to_delete:
                    temp.pop(e)
                max_idx = len(data[key]) - l
                for idx in range(round(max_idx)):
                    x_seq = np.asarray(temp)[idx:idx+l]
                    x_ret = []
                    for sample in x_seq:
                        for value in sample:
                            x_ret.append(value)
                    y_ret = Counter(np.asarray(label)[idx:idx+l]).most_common(1)[0][0]     
                    X.append(x_ret)
                    y.append(y_ret)  
            X = np.asarray(X)
            y = np.asarray(y)  
            df.append([l,m,X,y])
    index = [0,10,20,30,40]
    for i in index:
        df[i][1] = 'FULL'
    
    for i in range(50):
        print(df[i][2].shape)
    
    filename = 'sequence_dataset.pkl'
    with open(Path(ROOT_DIR) / f'progetto/finale/{filename}', 'wb') as f:
        dump(df, f)