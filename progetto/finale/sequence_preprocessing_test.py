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
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from sklearnex import patch_sklearn 

if __name__ == "__main__":
    filename = 'sequence_dataset.pkl'
    with open(Path(ROOT_DIR) / f'progetto/finale/{filename}','rb') as f:
        data = pickle.load(f)
    training = []
    test = []
    for i in range(50):
        ret1 = [data[i][0],data[i][1]]
        ret2 = [data[i][0],data[i][1]]
        train_size = round(len(data[i][2]) *0.8)
        test_size = round(len(data[i][2]) *0.2)
        ret1.append(data[i][2][:train_size])
        ret1.append(data[i][3][:train_size])
        ret2.append(data[i][2][train_size:train_size+test_size])
        ret2.append(data[i][3][train_size:train_size+test_size])
        training.append(ret1)
        test.append(ret2)

    data = [training,test]
      
    filename = 'sequence_dataset_test.pkl'
    with open(Path(ROOT_DIR) / f'progetto/finale/{filename}', 'wb') as f:
        dump(data, f)
        