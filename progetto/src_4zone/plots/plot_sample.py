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
    with open(Path(ROOT_DIR) / f'progetto/src_4zone/results/{filename}','rb') as f:
        result = pickle.load(f)

    data = []
    for key in result:
        k = key
        y = result[key] 
        data.append([k,y])
        
    
    modes_perf=[]   
    for i in range(21):
        temp = []
        y_true = data[i][1][1]
        y_pred = data[i][1][2]
        p = precision_score(y_true,y_pred,average=None).round(4)
        r = recall_score(y_true,y_pred,average=None).round(4)
        f = f1_score(y_true,y_pred,average=None).round(4)
        temp.append(p)
        temp.append(r)
        temp.append(f)
        modes_perf.append([data[i][1][0],temp])    
    conditions = ['FULL','VO2-VCO2','VE/VO2-VE/VCO2','RF-HR','Q-HR','Q-RF','HR']
    results = []

    for e in modes_perf:
        result = {
            'Features': e[0][1],
            'Classifier':  e[0][0],
            'Riposo': e[1][2][0],
            'Aerobica': e[1][2][1],
            'Anaerobica':e[1][2][2],
            'Defaticamento': e[1][2][3]
        }
        results.append(result)
    df = pd.DataFrame(results)
       
def create_and_save_plots():
    classifiers = df['Classifier'].unique()

    for classifier in classifiers:
        subset = df[df['Classifier'] == classifier]
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(subset['Features'], subset['Riposo'], marker='o', label='Riposo')
        plt.plot(subset['Features'], subset['Aerobica'], marker='o', label='Aerobica')
        plt.plot(subset['Features'], subset['Anaerobica'], marker='o', label='Anaerobica')
        plt.plot(subset['Features'], subset['Defaticamento'], marker='o', label='Defaticamento')
        plt.xlabel('Features')
        plt.ylabel('F1 Score')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.title(f'Andamento F1-Score per {classifier}')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(f'{classifier}_4zone_plot.png')
        plt.close()

# Call the function to create and save the plots
create_and_save_plots()