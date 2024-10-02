import os
import sys
sys.path.append(os.path.abspath(os.curdir).split('respirazione')[0] + 'respirazione')
from config.definitions import ROOT_DIR
from pathlib import Path
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearnex import patch_sklearn 

patch_sklearn()

if __name__ == "__main__":
    filename = 'dataset.pkl'
    with open(Path(ROOT_DIR) / f'progetto/src_4zone/{filename}','rb') as f:
        data = pickle.load(f)
    classifiers = [KNeighborsClassifier(),svm.SVC(),RandomForestClassifier()]
    
    modes = ['FULL','VO2-VCO2','VE/VO2-VE/VCO2','RF-HR','Q-HR','Q-RF','HR']
    params = [
        {
        'n_neighbors' : [1,5,10,20,50,100,200,300],
        'weights': ['uniform','distance'],
        },
        {
        'C': [1,10,100,1000],  
        'gamma': [1, 0.1, 0.01, 0.001], 
        'kernel': ['rbf']
        },
        {
        'max_features': ['sqrt', 'log2'], 
        'max_depth': [None,3,5,7], 
        'max_leaf_nodes': [None,3,6,9],
        'min_samples_split': [2,5,10,30,50]
        }
        ]
    scoring = ['accuracy','precision_macro','recall_macro','f1_macro']
    best = []
    for j in range(3):
        for i in range(7):
            print("\nTUNING:", classifiers[j], "WITH MODE: ",data[i][0])
            clf = GridSearchCV(classifiers[j],param_grid=params[j],scoring=scoring, refit='f1_macro')
            clf.fit(data[i][1],data[i][2])
            best.append([data[i][0],clf.best_params_,clf.best_score_])
    print(best)