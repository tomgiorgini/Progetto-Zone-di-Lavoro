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
    with open(Path(ROOT_DIR) / f'progetto/src_dati_puliti/results/{filename}','rb') as f:
        result = pickle.load(f)

    data = []
    for key in result:
        k = key
        y = result[key] 
        data.append([k,y])
    cmap = "Blues"
    labels = ['Riposo','Aerobica','Anaerobica']
    f, axes = plt.subplots(1, 1, figsize=(5,5))

    for i in range(1):
        for j in range(1):
            index = 10
            if index < len(data):
                axes.set_title('Random Forest, RF-HR')
                disp = ConfusionMatrixDisplay.from_predictions(data[index][1][1], data[index][1][2], display_labels=labels,ax=axes,cmap=cmap)

    plt.show()

    f.savefig("sample_predictor_results_3zone.png", bbox_inches='tight')