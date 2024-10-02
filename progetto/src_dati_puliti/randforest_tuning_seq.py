import os
import sys
sys.path.append(os.path.abspath(os.curdir).split('respirazione')[0] + 'respirazione')
from config.definitions import ROOT_DIR
from progetto.src_dati_puliti.sequence_predictor import sequence_create
from pathlib import Path
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearnex import patch_sklearn 

patch_sklearn()

if __name__ == "__main__":
    filename = 'sequence_dataset.pkl'
    with open(Path(ROOT_DIR) / f'progetto/dati_puliti/{filename}','rb') as f:
        out_data = pickle.load(f)
        
    modes = ['full','no-load','rf-hr']
    seq_len = [32]
    params = {
        'max_features': ['sqrt', 'log2'], 
        'max_depth': [None], 
        'max_leaf_nodes': [None],
        'min_samples_split': [2,3,4,5]
    }
    scoring = ['accuracy','precision_macro','recall_macro','f1_macro']
    rf = RandomForestClassifier()
    filename = 'rf_tuning.txt'
    for l in seq_len:
        print("\nTUNING RANDOM FOREST WITH SEQ_LEN = ",l)
        with open(Path(ROOT_DIR) / f'progetto/src_dati_puliti/output/{filename}',"a") as f:
                    print("\nTUNING RANDOM FOREST WITH SEQ_LEN = ",l,file=f)
        for m in modes:
            print("\nWITH MODE: ",m)
            with open(Path(ROOT_DIR) / f'progetto/src_dati_puliti/output/{filename}',"a") as f:
                    print("\nWITH MODE",m,file=f)
            X_train, X_test, y_train, y_test = sequence_create(out_data,m,l,30000,split=False)
            clf = GridSearchCV(rf,param_grid=params,scoring=scoring, refit='precision_macro')
            clf.fit(X_train,y_train)
            print("Migliori iperparametri:", clf.best_params_)
            print("Miglior score:", clf.best_score_)
            with open(Path(ROOT_DIR) / f'progetto/src_dati_puliti/output/{filename}',"a") as f:
                    print("Migliori iperparametri:", clf.best_params_,file=f)
                    print("Miglior score:", clf.best_score_,file=f)