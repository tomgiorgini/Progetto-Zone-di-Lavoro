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
    filename = 'sequence_predictor_results.pkl'
    with open(Path(ROOT_DIR) / f'progetto/src_3zone/results/{filename}','rb') as f:
        result = pickle.load(f)

    data = []
    for key in result:
        k = key
        y = result[key] 
        data.append([k,y])
    
    compareseq = []
    for j in range(3):
        temp = []
        for i in range(10):
            index = j*10 + i
            temp.append(data[index])
        compareseq.append(temp)
    subtitles = ['mode: full, seqlen = 2','mode: full, seqlen = 4','mode: full, seqlen = 8','mode: full, seqlen = 16','mode: full, seqlen = 32',
                 'mode: HR-RF, seqlen = 2','mode: HR-RF, seqlen = 4','mode: HR-RF, seqlen = 8','mode: HR-RF, seqlen = 16','mode: HR-RF, seqlen = 32']
    titles = ['Risultati del Sequence Predictor con KNeighbors, 3 zone','Risultati del Sequence Predictor con Random Forest, 3 zone','Risultati del Sequence Predictor con SVC, 3 zone']
    pngs = ['seq_predictor_results_3zone_knn.png','seq_predictor_results_3zone_rf.png','seq_predictor_results_3zone_svc.png']
    k = 0
    for e in compareseq:
        cmap = "Blues"
        labels = ['Riposo','Aerobica','Anaerobica']
        f, axes = plt.subplots(2, 5, figsize=(20, 10))

        for i in range(2):
            for j in range(5):
                index = i * 5 + j
                if index < len(e):
                    axes[i, j].set_title(subtitles[index])
                    disp = ConfusionMatrixDisplay.from_predictions(e[index][1][0], e[index][1][1], display_labels=labels,ax=axes[i, j],colorbar=False, cmap=cmap)
            
                    # Personalizzazione degli assi
                    if i != 1:  # rimuove i tick delle etichette x per tutte le righe eccetto l'ultima
                        #axes[i, j].xaxis.set_ticklabels(['', '', ''])
                        axes[i, j].set_xlabel('')
                        #axes[i, j].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                    if j != 0:  # rimuove i tick delle etichette y per tutte le colonne eccetto la prima
                        axes[i, j].yaxis.set_ticklabels(['', '', ''])
                        axes[i, j].set_ylabel('')
                        #axes[i, j].tick_params(axis='y', which='both', left=False)
                else:
                    axes[i, j].axis('off')  # Spegne gli assi per subplot non usati

        f.suptitle(titles[k], size=25, y=0.93)
        plt.show()
        
        f.savefig(pngs[k], bbox_inches='tight')
        k = k+1