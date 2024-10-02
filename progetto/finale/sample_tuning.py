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
from sklearn import svm

from sklearnex import patch_sklearn 

patch_sklearn()

if __name__ == "__main__":
    filename = 'dataset.pkl'
    with open(Path(ROOT_DIR) / f'progetto/finale/{filename}','rb') as f:
        data = pickle.load(f)
    rf = RandomForestClassifier()
    modes = ['FULL','VO2-VCO2','VE/VO2-VE/VCO2','RF-HR','Q-HR','Q-RF','Q-VO2','Q-VCO2','Q','HR']
    params = {
        'max_features': ['sqrt', 'log2'], 
        'max_depth': [None,3,5,7], 
        'max_leaf_nodes': [None,3,6,9],
        'min_samples_split': [2,5,10,30,50]
    }
    scoring = ['accuracy','precision_macro','recall_macro','f1_macro']
    for i in range(10):
        print("\nTUNING: Random Forest Classifier, WITH MODE: ",data[i][0])
        clf = GridSearchCV(rf,param_grid=params,scoring=scoring, refit='f1_macro')
        clf.fit(data[i][1],data[i][2])
        print("Migliori iperparametri:", clf.best_params_)
        print("Miglior score:", clf.best_score_)