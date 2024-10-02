import os
import sys
sys.path.append(os.path.abspath(os.curdir).split('respirazione')[0] + 'respirazione')
from config.definitions import ROOT_DIR
from pathlib import Path
import pickle
from sklearn.metrics import classification_report
from sklearn.preprocessing import label_binarize
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
    
    cmap = "Blues"
    labels = ['Riposo','Aerobica','Anaerobica']
    f, axes = plt.subplots(2, 5, figsize=(16, 8))

    for i in range(2):
        for j in range(5):
            index = i * 3 + j
            if index < len(data):
                axes[i, j].set_title(data[index][0])
                disp = ConfusionMatrixDisplay.from_predictions(data[index][1][0], data[index][1][1], display_labels=labels,ax=axes[i, j],colorbar=False, cmap=cmap)
                #disp.plot(include_values=True, cmap=cmap, ax=axes[i, j])


                # Personalizzazione degli assi
                if i != 1:  # rimuove i tick delle etichette x per tutte le righe eccetto l'ultima
                    axes[i, j].xaxis.set_ticklabels(['', '', ''])
                    axes[i, j].set_xlabel('')
                    axes[i, j].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                if j != 0:  # rimuove i tick delle etichette y per tutte le colonne eccetto la prima
                    axes[i, j].yaxis.set_ticklabels(['', '', ''])
                    axes[i, j].set_ylabel('')
                    axes[i, j].tick_params(axis='y', which='both', left=False)
            else:
                axes[i, j].axis('off')  # Spegne gli assi per subplot non usati

    f.suptitle("Risultati del Sample Predictor", size=20, y=0.93)
    plt.show()

    f.savefig("sample_predictor_confmat.pdf", bbox_inches='tight')
