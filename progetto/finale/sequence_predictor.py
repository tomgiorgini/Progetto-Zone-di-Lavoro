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


if __name__ == "__main__":
    filename = 'sequence_dataset_test.pkl'
    with open(Path(ROOT_DIR) / f'progetto/finale/{filename}','rb') as f:
        data = pickle.load(f)
    
    modes = ['FULL','VO2-VCO2','VE/VO2-VE/VCO2','RF-HR','Q-HR','Q-RF','Q-VO2','Q-VCO2','Q','HR']
    seq_len = [2,4,8,16,32]
    rf = RandomForestClassifier()
    target_names = ['Riposo','Aerobica','Anaerobica']
    results = dict()
    filename= "sequence_output_test.txt"
    with open(Path(ROOT_DIR) / f'progetto/finale/output/{filename}',"w") as f:
        print('',file=f)
    for i in range(50):
        with open(Path(ROOT_DIR) / f'progetto/finale/output/{filename}',"a") as f:
            print("Testing with seq_len: ",data[0][i][0],file=f)
        print("Testing with seq_len: ",data[0][i][0])
        with open(Path(ROOT_DIR) / f'progetto/finale/output/{filename}',"a") as f:
            print("with mode: ",data[0][i][1],file=f)
        print("with mode: ",data[0][i][1])
        start = timer()
        X_train=data[0][i][2] 
        X_test=data[1][i][2] 
        y_train=data[0][i][3] 
        y_test = data[1][i][3]
        clf = rf
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        clas = classification_report(y_true=y_test,y_pred=y_pred, target_names=target_names)
        key = str(data[0][i][0])+'-'+data[0][i][1]
        results[key] = [y_test,y_pred]
        end = timer()
        print(clas)
        print("elapsed time: ", end-start,"\n")
        with open(Path(ROOT_DIR) / f'progetto/finale/output/{filename}',"a") as f:
            print(clas,"elapsed time: ", end-start,'\n',file=f)
    filename = 'sequence_predictor_results_test.pkl'
    with open(Path(ROOT_DIR) / f'progetto/finale/results/{filename}', 'wb') as f:
        dump(results, f)