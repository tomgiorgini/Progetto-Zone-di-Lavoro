import os
import sys
sys.path.append(os.path.abspath(os.curdir).split('respirazione')[0] + 'respirazione')
from config.definitions import ROOT_DIR
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

if __name__ == "__main__":
    filename = 'sample_predictor_results.pkl'
    with open(Path(ROOT_DIR) / f'progetto/finale/results/{filename}','rb') as f:
        result = pickle.load(f)

    data = []
    for key in result:
        k = key
        y = result[key] 
        data.append([k,y])
        
    modes = ['FULL','VO2-VCO2','VE/VO2-VE/VCO2','RF-HR','Q-HR','Q-RF','Q-VO2','Q-VCO2','Q','HR']
        
    data_avg = []
    for i in range(10):
        res = []
        temp =  []
        y_true = data[i][1][0]
        y_pred = data[i][1][1]
        p = precision_score(y_true,y_pred,average='macro').round(4)
        r = recall_score(y_true,y_pred,average='macro').round(4)
        f = f1_score(y_true,y_pred,average='macro').round(4)
        temp.append(p)
        temp.append(r)
        temp.append(f)
        data_avg.append([data[i][0],temp])
    results = []
    for e in data_avg:
        result = {
                'Mode': e[0],
                'Precision': e[1][0],
                'Recall': e[1][1],
                'F1': e[1][2]
            }
        results.append(result)
    df_avg = pd.DataFrame(results)
    
    data_full = []
    for i in range(10):
        res = []
        temp =  []
        y_true = data[i][1][0]
        y_pred = data[i][1][1]
        p = precision_score(y_true,y_pred,average=None).round(4)
        r = recall_score(y_true,y_pred,average=None).round(4)
        f = f1_score(y_true,y_pred,average=None).round(4)
        temp.append(p)
        temp.append(r)
        temp.append(f)
        data_full.append([data[i][0],temp])
        
    results = []
    for e in data_full:
        result = {
                'Mode': e[0],
                'Riposo': e[1][0][0],
                'Aerobica': e[1][1][1],
                'Anaerobica': e[1][2][2]
            }
        results.append(result)
    df_full = pd.DataFrame(results)
# Creazione del subplot con 2 grafici affiancati
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# Primo grafico: Precision, Recall, F1-Score
axs[0].plot(df_avg['Mode'], df_avg['F1'], label='F1-Score', marker='o')

axs[0].set_xlabel('Features')
axs[0].set_ylabel('Score')
axs[0].set_title('Macro average f1-score')
axs[0].legend()
axs[0].tick_params(axis='x', rotation=45)

# Secondo grafico: Riposo, Aerobica, Anaerobica
axs[1].plot(df_full['Mode'], df_full['Riposo'], label='Riposo', marker='o')
axs[1].plot(df_full['Mode'], df_full['Aerobica'], label='Aerobica', marker='o')
axs[1].plot(df_full['Mode'], df_full['Anaerobica'], label='Anaerobica', marker='o')

axs[1].set_xlabel('Features')
axs[1].set_ylabel('Score')
axs[1].set_title('F1-score based on classification')
axs[1].legend()
axs[1].tick_params(axis='x', rotation=45)

# Mostrare il subplot

plt.show()
fig.savefig("sample_predictor_plot.pdf", bbox_inches='tight')


    