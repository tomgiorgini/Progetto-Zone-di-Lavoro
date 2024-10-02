import os
import sys
sys.path.append(os.path.abspath(os.curdir).split('respirazione')[0] + 'respirazione')
from config.definitions import ROOT_DIR
from progetto.src_3zone.sequence_predictor import sequence_create
from pathlib import Path
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from sklearnex import patch_sklearn 

patch_sklearn()

if __name__ == "__main__":
    filename = 'sequence_dataset.pkl'
    with open(Path(ROOT_DIR) / f'progetto/dati_originali/{filename}','rb') as f:
        out_data = pickle.load(f)
    
    modes = ['full','no-load','rf-hr']
    seq_len = [2,4,8,16,32]
    params = {
        'n_neighbors' : [1,5,10,20,50,100,200,300],
        'weights': ['uniform','distance'],
    }
    scoring = ['accuracy','precision_macro','recall_macro','f1_macro']
    filename = 'knn_tuning.txt'
    with open(Path(ROOT_DIR) / f'progetto/src_3zone/output/{filename}',"w") as f:
        print('',file=f)
    knn = KNeighborsClassifier()
    for l in seq_len:
        print("\nTUNING KNN WITH SEQ_LEN = ",l)
        with open(Path(ROOT_DIR) / f'progetto/src_3zone/output/{filename}',"a") as f:
                    print("\nTUNING KNN WITH SEQ_LEN = ",l,file=f)
        for m in modes:
            print("\nWITH MODE: ",m)
            with open(Path(ROOT_DIR) / f'progetto/src_3zone/output/{filename}',"a") as f:
                    print("\nWITH MODE",file=f)
            X_train, X_test, y_train, y_test = sequence_create(out_data,m,l,50000,split=False)
            clf = GridSearchCV(knn,param_grid=params,scoring=scoring, refit='precision_macro')
            clf.fit(X_train,y_train)
            print("Migliori iperparametri:", clf.best_params_)
            print("Miglior score:", clf.best_score_)
            with open(Path(ROOT_DIR) / f'progetto/src_3zone/output/{filename}',"a") as f:
                    print("Migliori iperparametri:", clf.best_params_,file=f)
                    print("Miglior score:", clf.best_score_,file=f)