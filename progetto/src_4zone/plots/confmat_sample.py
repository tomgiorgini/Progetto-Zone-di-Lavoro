import os
import sys
sys.path.append(os.path.abspath(os.curdir).split('respirazione')[0] + 'respirazione')
from config.definitions import ROOT_DIR
from pathlib import Path
import pickle
from sklearn.metrics import classification_report
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
    cmap = "Blues"
    labels = ['Riposo','Aerobica','Anaerob','Def']
    f, axes = plt.subplots(3, 7, figsize=(27, 19))

    for i in range(3):
        for j in range(7):
            index = i * 7 + j
            if index < len(data):
                axes[i, j].set_title(data[index][0])
                disp = ConfusionMatrixDisplay.from_predictions(data[index][1][1], data[index][1][2], display_labels=labels,ax=axes[i, j],colorbar=False,cmap=cmap)
            
                # Personalizzazione degli assi
                if i != 2:  # rimuove i tick delle etichette x per tutte le righe eccetto l'ultima
                    #axes[i, j].xaxis.set_ticklabels(['', '', ''])
                    axes[i, j].set_xlabel('')
                    #axes[i, j].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                if j != 0:  # rimuove i tick delle etichette y per tutte le colonne eccetto la prima
                    axes[i, j].yaxis.set_ticklabels(['', '', '',''])
                    axes[i, j].set_ylabel('')
                    axes[i, j].tick_params(axis='y', which='both', left=False)
            else:
                axes[i, j].axis('off')  # Spegne gli assi per subplot non usati

    f.suptitle("Risultati dei Sample Predictor, con 4 zone", size=20, y=0.93)

        
    f.savefig("sample_predictor_results_4zone.png", bbox_inches='tight')