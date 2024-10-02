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
from sklearn import svm
from timeit import default_timer as timer
from collections import Counter
from sklearnex import patch_sklearn 

patch_sklearn()

def sequence_create(datain,mode,seq_len,n_samples,split = True):
    X=[]
    y=[]
    data = copy.deepcopy(datain)
    l = len(data.keys())
    samplesxset = round(n_samples /l) # calcolo il numero di sequenze da campionare per ogni set di sample
    if mode == 'full': # all features
        for key in data:
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
    with open(Path(ROOT_DIR) / f'progetto/dati_puliti/{filename}','rb') as f:
        out_data = pickle.load(f)
        
    modes = ['full','no-load','rf-hr']
    models = ["KNeighborsClassifier","RandomForestClassifier","SVClassifier"]
    
    knn_params = [{'n_neighbors': 1, 'weights': 'uniform'},
                  {'n_neighbors': 5, 'weights': 'distance'},
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
                  {'C': 100, 'gamma': 0.1},
                  {'C': 10, 'gamma': 1},
                  {'C': 10, 'gamma': 0.1},
                  {'C': 1, 'gamma': 0.1},
                  {'C': 100, 'gamma': 0.001},
                  {'C': 1, 'gamma': 1}
    ]
    # ho un set di classificatori per ogni seq_len, ognuno con i suoi iperparametri migliori
    # calcolati nel tuning
    classifiers = [
                  [[KNeighborsClassifier(**(knn_params[0])),KNeighborsClassifier(**(knn_params[0])),KNeighborsClassifier(**(knn_params[3]))],
                   [RandomForestClassifier(**(rf_params[3])),RandomForestClassifier(**(rf_params[3])),RandomForestClassifier(**(rf_params[4]))],
                   [svm.SVC(**(svc_params[1])),svm.SVC(**(svc_params[3])),svm.SVC(**(svc_params[0]))]],
                  
                  [[KNeighborsClassifier(**(knn_params[0])),KNeighborsClassifier(**(knn_params[0])),KNeighborsClassifier(**(knn_params[4]))],
                   [RandomForestClassifier(**(rf_params[2])),RandomForestClassifier(**(rf_params[3])),RandomForestClassifier(**(rf_params[3]))],
                   [svm.SVC(**(svc_params[2])),svm.SVC(**(svc_params[5])),svm.SVC(**(svc_params[6]))]],
                  
                  [[KNeighborsClassifier(**(knn_params[0])),KNeighborsClassifier(**(knn_params[0])),KNeighborsClassifier(**(knn_params[1]))],
                   [RandomForestClassifier(**(rf_params[4])),RandomForestClassifier(**(rf_params[4])),RandomForestClassifier(**(rf_params[4]))],
                   [svm.SVC(**(svc_params[2])),svm.SVC(**(svc_params[6])),svm.SVC(**(svc_params[7]))]],
                  
                  [[KNeighborsClassifier(**(knn_params[0])),KNeighborsClassifier(**(knn_params[0])),KNeighborsClassifier(**(knn_params[0]))],
                   [RandomForestClassifier(**(rf_params[3])),RandomForestClassifier(**(rf_params[4])),RandomForestClassifier(**(rf_params[3]))],
                   [svm.SVC(**(svc_params[2])),svm.SVC(**(svc_params[5])),svm.SVC(**(svc_params[6]))]],
                  
                  [[KNeighborsClassifier(**(knn_params[0])),KNeighborsClassifier(**(knn_params[0])),KNeighborsClassifier(**(knn_params[0]))],
                   [RandomForestClassifier(**(rf_params[5])),RandomForestClassifier(**(rf_params[4])),RandomForestClassifier(**(rf_params[5]))],
                   [svm.SVC(**(svc_params[9])),svm.SVC(**(svc_params[4])),svm.SVC(**(svc_params[6]))]]
    ]
    
    target_names = ['Riposo','Aerobica','Anaerobica']
    filename= "output_seq.txt"
    results = dict()
    seq_len = [2,4,8,16,32]
    with open(Path(ROOT_DIR) / f'progetto/src_dati_puliti/output/{filename}',"w") as f:
        print('',file=f)
    for i in range(3):
        with open(Path(ROOT_DIR) / f'progetto/src_dati_puliti/output/{filename}',"a") as f:
            print('\nTESTING ',models[i],file=f)
        print("\nTESTING ",models[i])
        for j in range(3):
            with open(Path(ROOT_DIR) / f'progetto/src_dati_puliti/output/{filename}',"a") as f:
                print("mode: ",modes[j],file=f)
            print("mode: ",modes[j])
            k = 0
            for l in seq_len:
                with open(Path(ROOT_DIR) / f'progetto/src_dati_puliti/output/{filename}',"a") as f:
                    print("with seq_len = ",l,file=f)
                print("with seq_len = ",l)
                start = timer()
                # il numero di sequenze è arbitrario
                X_train, X_test, y_train, y_test = sequence_create(out_data,modes[j],l,30000)
                clf = classifiers[k][i][j]
                clf.fit(X_train,y_train)
                y_pred = clf.predict(X_test)
                clas = classification_report(y_true=y_test,y_pred=y_pred, target_names=target_names)
                lstr = str(l)
                key = models[i]+'-'+lstr+'-'+modes[j]
                results[key] = [y_test,y_pred]
                end = timer()
                print(clas)
                print("elapsed time: ", end-start,"\n")
                with open(Path(ROOT_DIR) / f'progetto/src_dati_puliti/output/{filename}',"a") as f:
                    print(clas,"elapsed time: ", end-start,'\n',file=f)
                k = k+1
    filename = 'sequence_predictor_results.pkl'
    with open(Path(ROOT_DIR) / f'progetto/src_dati_puliti/results/{filename}', 'wb') as f:
        dump(results, f)