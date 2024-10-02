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
    filename = 'sequence_dataset.pkl'
    with open(Path(ROOT_DIR) / f'progetto/src_3zone/{filename}','rb') as f:
        data = pickle.load(f)
    modes = ['FULL','RF-HR']
    seq_len = [2,4,8,16,32]
    
    models = ["KNeighborsClassifier","RandomForestClassifier","SVClassifier"]
    
    knn_params = [{'n_neighbors': 1, 'weights': 'uniform'},
                  {'n_neighbors': 5, 'weights': 'uniform'},
                  {'n_neighbors': 10, 'weights': 'distance'},
                  {'n_neighbors': 20, 'weights': 'distance'},
                  {'n_neighbors': 50, 'weights': 'distance'}      
    ]
    rf_params = [{'n_estimators': 500,'max_depth': None, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'min_samples_split': 5},
                 {'n_estimators': 500, 'max_depth': None, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'min_samples_split': 4},
                 {'n_estimators': 500,'max_depth': None, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'min_samples_split': 3},
                 {'n_estimators': 500,'max_depth': None, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'min_samples_split': 2},
                 {'n_estimators': 500,'max_depth': None, 'max_features': 'log2', 'max_leaf_nodes': None, 'min_samples_split': 2},
                 {'n_estimators': 500,'max_depth': None, 'max_features': 'log2', 'max_leaf_nodes': None, 'min_samples_split': 3},
                 {'n_estimators': 500,'max_depth': None, 'max_features': 'log2', 'max_leaf_nodes': None, 'min_samples_split': 4}             
    ]
    svc_params = [{'C': 1000, 'gamma': 1},
                  {'C': 1000, 'gamma': 0.01},
                  {'C': 1000, 'gamma': 0.001},
                  {'C': 100, 'gamma': 1},
                  {'C': 100, 'gamma': 0.01},
                  {'C': 10, 'gamma': 1,},
                  {'C': 10, 'gamma': 0.1},
                  {'C': 1, 'gamma': 0.1},
                  {'C': 100, 'gamma': 0.001},
                  {'C': 1, 'gamma': 1}
    ]
    classifiers = [
                  [[KNeighborsClassifier(**(knn_params[0])),KNeighborsClassifier(**(knn_params[4]))],
                   [RandomForestClassifier(**(rf_params[5])),RandomForestClassifier(**(rf_params[1]))],
                   [svm.SVC(**(svc_params[5])),svm.SVC(**(svc_params[6]))]],
                  
                  [[KNeighborsClassifier(**(knn_params[0])),KNeighborsClassifier(**(knn_params[3]))],
                   [RandomForestClassifier(**(rf_params[2])),RandomForestClassifier(**(rf_params[2]))],
                   [svm.SVC(**(svc_params[0])),svm.SVC(**(svc_params[5]))]],
                  
                  [[KNeighborsClassifier(**(knn_params[0])),KNeighborsClassifier(**(knn_params[2]))],
                   [RandomForestClassifier(**(rf_params[4])),RandomForestClassifier(**(rf_params[3]))],
                   [svm.SVC(**(svc_params[5])),svm.SVC(**(svc_params[9]))]],
                  
                  [[KNeighborsClassifier(**(knn_params[0])),KNeighborsClassifier(**(knn_params[0]))],
                   [RandomForestClassifier(**(rf_params[3])),RandomForestClassifier(**(rf_params[2]))],
                   [svm.SVC(**(svc_params[6])),svm.SVC(**(svc_params[5]))]],
                  
                  [[KNeighborsClassifier(**(knn_params[0])),KNeighborsClassifier(**(knn_params[0]))],
                   [RandomForestClassifier(**(rf_params[4])),RandomForestClassifier(**(rf_params[5]))],
                   [svm.SVC(**(svc_params[6])),svm.SVC(**(svc_params[5]))]]
    ]
    
    classi = [KNeighborsClassifier(), RandomForestClassifier(), svm.SVC()]
    
    target_names = ['Riposo','Aerobica','Anaerobica']
    results = dict()
    filename= "sequence_output.txt"
    with open(Path(ROOT_DIR) / f'progetto/src_3zone/output/{filename}',"w") as f:
        print('',file=f)
    for i in range(10):
        with open(Path(ROOT_DIR) / f'progetto/src_3zone/output/{filename}',"a") as f:
            print("Testing with seq_len: ",data[0][i][0],file=f)
        print("Testing with seq_len: ",data[0][i][0])
        with open(Path(ROOT_DIR) / f'progetto/src_3zone/output/{filename}',"a") as f:
            print("with mode: ",data[0][i][1],file=f)
        print("with mode: ",data[0][i][1])
        start = timer()
        X_train=data[0][i][2] 
        X_test=data[1][i][2] 
        y_train=data[0][i][3] 
        y_test = data[1][i][3]
        for j in range(3):
            print("with classfier ", classi[j],"\n")
            with open(Path(ROOT_DIR) / f'progetto/src_3zone/output/{filename}',"a") as f:
                print("with classfier ", classi[j],"\n",file=f)
            clf = classi[j]
            clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            clas = classification_report(y_true=y_test,y_pred=y_pred, target_names=target_names)
            key = models[j]+'-'+str(data[0][i][0])+'-'+data[0][i][1]
            results[key] = [y_test,y_pred]
            end = timer()
            print(clas)
            print("elapsed time: ", end-start,"\n")
            with open(Path(ROOT_DIR) / f'progetto/src_3zone/output/{filename}',"a") as f:
                print(clas,"elapsed time: ", end-start,'\n',file=f)
    filename = 'sequence_predictor_results.pkl'
    with open(Path(ROOT_DIR) / f'progetto/src_3zone/results/{filename}', 'wb') as f:
        dump(results, f)