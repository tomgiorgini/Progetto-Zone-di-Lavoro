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
import seaborn as sns

if __name__ == "__main__":
    filename = 'sequence_predictor_results.pkl'
    with open(Path(ROOT_DIR) / f'progetto/src_dati_puliti/results/{filename}','rb') as f:
        result = pickle.load(f)

    data = []
    for key in result:
        k = key
        y = result[key] 
        data.append([k,y])
        
    perf = []    
    for e in data:
        res = []
        y_true = e[1][0]
        y_pred = e[1][1]
        f = f1_score(y_true,y_pred,average=None).round(4)
        res.append([e[0],f])
        perf.append(res)
results = []

for item in perf:
    name, metrics = item[0]

   
    param, mode = name.split('-',1)
    
    results.append({
        'Mode': mode,
        'Seq Len': param,
        'Riposo': metrics[0],
        'Aerobica': metrics[1],
        'Anaerobica': metrics[2],
    })


df = pd.DataFrame(results)

# Raggruppiamo i dati per 'Mode' e 'Seq Len'
modes = ["FULL", "RF-HR"]
seq_lens = [2, 4, 8, 16, 32]
classifiers = ["RandomForestClassifier"]

# Creare la figura e gli assi (2 righe, 5 colonne)
fig, axes = plt.subplots(2, 1, figsize=(10, 25))

# Iterare su ogni modalità e lunghezza di sequenza per creare i grafici con andamento
for j, mode in enumerate(modes):
    for i, classifier in enumerate(classifiers):
        # Filtriamo i dati per la modalità e la lunghezza di sequenza corrente
        subset = df[(df["Mode"] == mode)]
        
        # Impostare l'asse corrente
        ax = axes[j]
        
        # Tracciare l'andamento delle metriche
        ax.plot(subset["Seq Len"], subset["Riposo"], marker='o', label='Riposo')
        ax.plot(subset["Seq Len"], subset["Aerobica"], marker='o', label='Aerobica')
        ax.plot(subset["Seq Len"], subset["Anaerobica"], marker='o', label='Anaerobica')
        
        # Impostare il titolo del grafico
        ax.set_title(f"{classifier} - {mode} ")
        ax.set_xlabel("Classifier")
        ax.set_ylabel("F1 score")
        ax.legend(loc='upper left')
        ax.grid(True)

# Regolare lo spazio tra i grafici
plt.show()

fig.savefig("seq_predictor_dati_puliti_plot.png")