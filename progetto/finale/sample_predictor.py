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
import matplotlib.pyplot as plt
patch_sklearn()


if __name__ == "__main__":
    filename = 'dataset.pkl'
    with open(Path(ROOT_DIR) / f'progetto/finale/{filename}','rb') as f:
        data = pickle.load(f)
        

                
    modes = ['FULL','VO2-VCO2','VE/VO2-VE/VCO2','RF-HR','Q-HR','Q-RF','Q-VO2','Q-VCO2','Q','HR']
    params = [{'max_depth': None, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'min_samples_split': 2},
              {'max_depth': None, 'max_features': 'log2', 'max_leaf_nodes': None, 'min_samples_split': 30},
              {'max_depth': None, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'min_samples_split': 20},
              {'max_depth': None, 'max_features': 'log2', 'max_leaf_nodes': None, 'min_samples_split': 15},
              {'max_depth': None, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'min_samples_split': 15},
              {'max_depth': None, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'min_samples_split': 30},
              {'max_depth': None, 'max_features': 'log2', 'max_leaf_nodes': None, 'min_samples_split': 30},
              {'max_depth': None, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'min_samples_split': 15},
              {'max_depth': None, 'max_features': 'log2', 'max_leaf_nodes': None, 'min_samples_split': 50},
              {'max_depth': None, 'max_features': 'log2', 'max_leaf_nodes': None, 'min_samples_split': 2}
              ]
    classifiers = [RandomForestClassifier(**(params[0])),RandomForestClassifier(**(params[1])),RandomForestClassifier(**(params[2])),
                   RandomForestClassifier(**(params[3])),RandomForestClassifier(**(params[4])),RandomForestClassifier(**(params[5])),
                   RandomForestClassifier(**(params[6])),RandomForestClassifier(**(params[7])),RandomForestClassifier(**(params[8])),
                   RandomForestClassifier(**(params[9]))]
    
    target_names = ['Riposo','Aerobica','Anaerobica']
    results = dict()
    filename= "output.txt"
    
    
    with open(Path(ROOT_DIR) / f'progetto/finale/output/{filename}',"w") as f:
        print('',file=f)
    for i in range(10):
        with open(Path(ROOT_DIR) / f'progetto/finale/output/{filename}',"a") as f:
            print("Testing with mode: ",modes[i],file=f)
        print("Testing with mode: ",modes[i])
        start = timer()
        X_train, X_test, y_train, y_test = train_test_split(data[i][1],data[i][2],test_size=0.20,random_state=42)
        clf = classifiers[i]
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        clas = classification_report(y_true=y_test,y_pred=y_pred, target_names=target_names)
        key = modes[i]
        results[key] = [y_test,y_pred]
        end = timer()
        print(clas)
        print("elapsed time: ", end-start,"\n")
        with open(Path(ROOT_DIR) / f'progetto/finale/output/{filename}',"a") as f:
            print(clas,"elapsed time: ", end-start,'\n',file=f)
    print(results)
    filename = 'sample_predictor_results.pkl'
    with open(Path(ROOT_DIR) / f'progetto/finale/results/{filename}', 'wb') as f:
        dump(results, f)
    