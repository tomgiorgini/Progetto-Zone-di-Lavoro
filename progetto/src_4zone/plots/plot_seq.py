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
    with open(Path(ROOT_DIR) / f'progetto/src_4zone/results/{filename}','rb') as f:
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

   
    classifier, param, mode = name.split('-',2)
    
    results.append({
        'Classifier': classifier,
        'Mode': mode,
        'Seq Len': param,
        'Riposo': metrics[0],
        'Aerobica': metrics[1],
        'Anaerobica': metrics[2],
        'Defaticamento': metrics[3]
    })


df = pd.DataFrame(results)

import matplotlib.pyplot as plt
import seaborn as sns

# Raggruppiamo i dati per 'Mode' e 'Seq Len'
modes = ["FULL", "RF-HR"]
seq_lens = [2, 4, 8, 16, 32]
classifiers = ["KNeighborsClassifier", "RandomForestClassifier", "SVClassifier"]

# Creare la figura e gli assi (2 righe, 5 colonne)
fig, axes = plt.subplots(2, 3, figsize=(20, 10))

# Iterare su ogni modalità e lunghezza di sequenza per creare i grafici con andamento
for i, mode in enumerate(modes):
    for j, classifier in enumerate(classifiers):
        # Filtriamo i dati per la modalità e la lunghezza di sequenza corrente
        subset = df[(df["Mode"] == mode) & (df["Classifier"] == classifier)]
        
        # Impostare l'asse corrente
        ax = axes[i, j]
        
        # Tracciare l'andamento delle metriche
        ax.plot(subset["Seq Len"], subset["Riposo"], marker='o', label='Riposo')
        ax.plot(subset["Seq Len"], subset["Aerobica"], marker='o', label='Aerobica')
        ax.plot(subset["Seq Len"], subset["Anaerobica"], marker='o', label='Anaerobica')
        ax.plot(subset["Seq Len"], subset["Defaticamento"], marker='o', label='Defaticamento')
        
        # Impostare il titolo del grafico
        ax.set_title(f"Features: {mode}, Classifier: {classifier}")
        ax.set_xlabel("Classifier")
        ax.set_ylabel("F1 score")
        ax.legend(loc='upper left')

# Regolare lo spazio tra i grafici
plt.tight_layout()
plt.show()

fig.savefig("seq_predictor_4zone_plot.png")