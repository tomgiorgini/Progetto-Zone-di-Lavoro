import os
import sys
sys.path.append(os.path.abspath(os.curdir).split('respirazione')[0] + 'respirazione')
from config.definitions import ROOT_DIR
from pathlib import Path
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
from sklearnex import patch_sklearn 

patch_sklearn()

if __name__ == "__main__":
    filename = 'dataset.pkl'
    with open(Path(ROOT_DIR) / f'progetto/src_dati_puliti/{filename}','rb') as f:
        data = pickle.load(f)
        
    for e in data:
        for v in e[1]:
            v[v>1e308] = 0
            v[np.isnan(v)] = 0
    #TESTING CLASSIFIERS
    modes = ['FULL','VO2-VCO2','VE/VO2-VE/VCO2','RF-HR','Q-HR','Q-RF','HR']
    models = ["KNN","RF","SVC"]

    params = [
            [{'n_neighbors': 1, 'weights': 'uniform'},
            {'n_neighbors': 10, 'weights': 'uniform'},
            {'n_neighbors': 50, 'weights': 'distance'},
            {'n_neighbors': 300, 'weights': 'distance'}
            ],
            [{'max_depth': None, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'min_samples_split': 2},
            {'max_depth': None, 'max_features': 'log2', 'max_leaf_nodes': None, 'min_samples_split': 30},
            {'max_depth': None, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'min_samples_split': 20},
            {'max_depth': None, 'max_features': 'log2', 'max_leaf_nodes': None, 'min_samples_split': 15},
            {'max_depth': None, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'min_samples_split': 15},
            {'max_depth': None, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'min_samples_split': 30},
            {'max_depth': None, 'max_features': 'log2', 'max_leaf_nodes': None, 'min_samples_split': 2}
            ],
            [{'C': 1000, 'gamma': 0.1},
            {'C': 1000, 'gamma': 1},
            {'C': 100, 'gamma': 1},
            {'C': 100, 'gamma': 0.001}
            ]
            ]
    classifiers = [[KNeighborsClassifier(**(params[0][0])),KNeighborsClassifier(**(params[0][1])),KNeighborsClassifier(**(params[0][2])),
                    KNeighborsClassifier(**(params[0][1])),KNeighborsClassifier(**(params[0][1])),KNeighborsClassifier(**(params[0][1])),
                    KNeighborsClassifier(**(params[0][3]))],
                  [RandomForestClassifier(**(params[1][0])),RandomForestClassifier(**(params[1][1])),RandomForestClassifier(**(params[1][2])),
                   RandomForestClassifier(**(params[1][3])),RandomForestClassifier(**(params[1][4])),RandomForestClassifier(**(params[1][5])),
                   RandomForestClassifier(**(params[1][6]))],
                  [svm.SVC(**(params[2][1])),svm.SVC(**(params[2][1])),svm.SVC(**(params[2][0])),svm.SVC(**(params[2][2])),
                   svm.SVC(**(params[2][1])),svm.SVC(**(params[2][2])),svm.SVC(**(params[2][3]))]
                   ]
    target_names = ['Riposo','Aerobica','Anaerobica']
    results = dict()
    filename= "output.txt"
    with open(Path(ROOT_DIR) / f'progetto/src_dati_puliti/output/{filename}',"w") as f:
        print('',file=f)
    for j in range(3):
        with open(Path(ROOT_DIR) / f'progetto/src_dati_puliti/output/{filename}',"a") as f:
            print("Testing with Classifier: ",models[j],file=f)
        print("Testing with Classifier: ",models[j])
        for i in range(7):
            with open(Path(ROOT_DIR) / f'progetto/src_dati_puliti/output/{filename}',"a") as f:
                print("with mode: ",modes[i],file=f)
            print("with mode: ",modes[i])
            start = timer()
            X_train, X_test, y_train, y_test = train_test_split(data[i][1],data[i][2],test_size=0.20,random_state=42)
            clf = classifiers[j][i]
            clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            clas = classification_report(y_true=y_test,y_pred=y_pred, target_names=target_names)
            key = models[j] + '-'+ modes[i]
            temp = [models[j],modes[i]]
            results[key] = [temp,y_test,y_pred]
            end = timer()
            print(clas)
            print("elapsed time: ", end-start,"\n")
            with open(Path(ROOT_DIR) / f'progetto/src_dati_puliti/output/{filename}',"a") as f:
                print(clas,"elapsed time: ", end-start,'\n',file=f)
        filename = 'sample_predictor_results.pkl'
        with open(Path(ROOT_DIR) / f'progetto/src_dati_puliti/results/{filename}', 'wb') as f:
            dump(results, f)
    