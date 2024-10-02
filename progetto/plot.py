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
from sklearn.preprocessing import MinMaxScaler

if __name__ == "__main__":
    # Data for the table extracted from the text
    data = {
    "Classifier": ["KNN", "KNN", "KNN", "KNN", "KNN", "KNN", "KNN",
                   "RF", "RF", "RF", "RF", "RF", "RF", "RF",
                   "SVC", "SVC", "SVC", "SVC", "SVC", "SVC", "SVC"],
    "Mode": ["FULL", "VO2-VCO2", "VE/VO2-VE/VCO2", "RF-HR", "Q-HR", "Q-RF", "HR",
             "FULL", "VO2-VCO2", "VE/VO2-VE/VCO2", "RF-HR", "Q-HR", "Q-RF", "HR",
             "FULL", "VO2-VCO2", "VE/VO2-VE/VCO2", "RF-HR", "Q-HR", "Q-RF", "HR"],
    "Precision_Riposo": [0.98, 0.82, 0.77, 0.68, 0.81, 0.78, 0.72, 0.98, 0.82, 0.77, 0.68, 0.81, 0.78, 0.72,
                         0.98, 0.82, 0.77, 0.68, 0.81, 0.78, 0.72],
    "Recall_Riposo": [0.99, 0.88, 0.84, 0.72, 0.85, 0.85, 0.66, 0.99, 0.88, 0.84, 0.72, 0.85, 0.85, 0.66,
                      0.99, 0.88, 0.84, 0.72, 0.85, 0.85, 0.66],
    "F1_Riposo": [0.99, 0.85, 0.81, 0.70, 0.83, 0.81, 0.69, 0.99, 0.85, 0.81, 0.70, 0.83, 0.81, 0.69,
                  0.99, 0.85, 0.81, 0.70, 0.83, 0.81, 0.69],
    "Precision_Aerobica": [0.95, 0.60, 0.63, 0.46, 0.62, 0.52, 0.46, 0.95, 0.60, 0.63, 0.46, 0.62, 0.52, 0.46,
                           0.95, 0.60, 0.63, 0.46, 0.62, 0.52, 0.46],
    "Recall_Aerobica": [0.97, 0.56, 0.53, 0.34, 0.54, 0.42, 0.29, 0.97, 0.56, 0.53, 0.34, 0.54, 0.42, 0.29,
                        0.97, 0.56, 0.53, 0.34, 0.54, 0.42, 0.29],
    "F1_Aerobica": [0.96, 0.58, 0.57, 0.39, 0.58, 0.47, 0.36, 0.96, 0.58, 0.57, 0.39, 0.58, 0.47, 0.36,
                    0.96, 0.58, 0.57, 0.39, 0.58, 0.47, 0.36],
    "Precision_Anaerobica": [1.00, 0.90, 0.91, 0.74, 0.87, 0.86, 0.69, 1.00, 0.90, 0.91, 0.74, 0.87, 0.86, 0.69,
                             1.00, 0.90, 0.91, 0.74, 0.87, 0.86, 0.69],
    "Recall_Anaerobica": [0.99, 0.88, 0.91, 0.78, 0.88, 0.88, 0.80, 0.99, 0.88, 0.91, 0.78, 0.88, 0.88, 0.80,
                          0.99, 0.88, 0.91, 0.78, 0.88, 0.88, 0.80],
    "F1_Anaerobica": [0.99, 0.89, 0.91, 0.76, 0.88, 0.87, 0.74, 0.99, 0.89, 0.91, 0.76, 0.88, 0.87, 0.74,
                      0.99, 0.89, 0.91, 0.76, 0.88, 0.87, 0.74],
    "Accuracy": [0.99, 0.83, 0.83, 0.69, 0.82, 0.79, 0.67, 0.99, 0.83, 0.83, 0.69, 0.82, 0.79, 0.67,
                 0.99, 0.83, 0.83, 0.69, 0.82, 0.79, 0.67],
    "Elapsed Time (s)": [0.597, 0.384, 0.133, 0.410, 0.360, 0.375, 0.489, 0.597, 0.384, 0.133, 0.410, 0.360, 0.375, 0.489,
                         0.597, 0.384, 0.133, 0.410, 0.360, 0.375, 0.489]
}

# Create a DataFrame
df = pd.DataFrame(data)

df_f1 = df[["Classifier", "Mode", "F1_Riposo", "F1_Aerobica", "F1_Anaerobica"]]
df_f1["Classifier"] = df_f1["Classifier"].replace({"KNN": "KNeighbours", "RF": "Random Forest"})

df_f1 = df_f1.rename(columns={"Mode": "Features"})

# Splitting into separate tables based on classifier
kneighbours_df = df_f1[df_f1["Classifier"] == "KNeighbours"]
random_forest_df = df_f1[df_f1["Classifier"] == "Random Forest"]
svc_df = df_f1[df_f1["Classifier"] == "SVC"]

# Function to create and plot the F1-score metrics for each classifier
def plot_f1_scores(df, classifier_name):
    features = df["Features"]
    f1_riposo = df["F1_Riposo"]
    f1_aerobica = df["F1_Aerobica"]
    f1_anaerobica = df["F1_Anaerobica"]

    plt.figure(figsize=(10, 6))
    plt.plot(features, f1_riposo, label="F1 Riposo", marker='o')
    plt.plot(features, f1_aerobica, label="F1 Aerobica", marker='o')
    plt.plot(features, f1_anaerobica, label="F1 Anaerobica", marker='o')
    plt.title(f'F1-Scores per Class for {classifier_name}')
    plt.xlabel("Features")
    plt.ylabel("F1-Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plotting for each classifier
plot_f1_scores(kneighbours_df, "KNeighbours")
plot_f1_scores(random_forest_df, "Random Forest")
plot_f1_scores(svc_df, "SVC")