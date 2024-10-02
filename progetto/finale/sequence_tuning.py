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
from scipy.stats import randint
from sklearnex import patch_sklearn 

patch_sklearn()

if __name__ == "__main__":
    filename = 'sequence_dataset.pkl'
    with open(Path(ROOT_DIR) / f'progetto/finale/{filename}','rb') as f:
        data = pickle.load(f)
    rf = RandomForestClassifier()
    modes = ['FULL','VO2-VCO2','VE/VO2-VE/VCO2','RF-HR','Q-HR','Q-RF','Q-VO2','Q-VCO2','Q','HR']
    param_dist = {
    'n_estimators': randint(100, 1000),  # Numero di alberi tra 100 e 1000
    'max_depth': [None, 10, 20, 30, 40, 50],  # Profondità massima degli alberi
    'min_samples_split': randint(2, 20),  # Minimo campioni per dividere un nodo
    'min_samples_leaf': randint(1, 10),  # Minimo campioni nelle foglie
    'max_features': ['sqrt', 'log2'],  # Numero massimo di feature considerate per ogni divisione
}

    scoring = ['accuracy','precision_macro','recall_macro','f1_macro']
    for i in range(50):
        print("\nTUNING: Random Forest Classifier, with seq_len: ",data[i][0],'and mode: ', data[i][1])
        clf = RandomizedSearchCV(estimator=rf, 
                                   param_distributions=param_dist, 
                                   n_iter=10,  # Numero di iterazioni casuali da provare
                                   cv=5, 
                                   scoring=scoring, 
                                   n_jobs=-1, 
                                   verbose=2, 
                                   random_state=42,refit='f1_macro')
        clf.fit(data[i][2],data[i][3])
        print("Migliori iperparametri:", clf.best_params_)
        print("Miglior score:", clf.best_score_)