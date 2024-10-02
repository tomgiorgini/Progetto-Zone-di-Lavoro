import os
import sys
sys.path.append(os.path.abspath(os.curdir).split('respirazione')[0] + 'respirazione')
from config.definitions import ROOT_DIR
from pathlib import Path
import pandas as pd
from datetime import datetime
import numpy as np

def load_and_convert(cpet_data_folder):
    if isinstance(cpet_data_folder, str):
        cpet_data_folder = Path(cpet_data_folder)
    files = sorted(os.listdir(cpet_data_folder))
    out_data = dict()
    for file_path in files:
        if not (file_path.split('.')[1] == 'xlsx'):
            continue
        xls = pd.ExcelFile(cpet_data_folder / file_path)
        # Load each sheet into a DataFrame
        test_df = pd.read_excel(xls, 'Test')
        at_df = pd.read_excel(xls, 'AT')
        rc_df = pd.read_excel(xls, 'RC')
        media_rest_df = pd.read_excel(xls, 'Media Rest')
        df_list = [test_df, at_df, rc_df, media_rest_df]
        for df in df_list:
            df.rename(columns={'Power': 'Load'}, inplace=True)
            df.rename(columns={'Rf': 'RF'}, inplace=True)
        
        # Remove unit rows and convert columns to appropriate datatypes for normalization
        test_df = test_df.drop([0, 1]).reset_index(drop=True)
        media_rest_df = media_rest_df.drop([0]).reset_index(drop=True)
        # Convert all columns except 't' to numeric
        numeric_cols = test_df.columns.drop('t')
        test_df['t'] = [datetime.combine(datetime.min, t) - datetime.min for t in test_df['t']]
        test_df[numeric_cols] = test_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        media_rest_df[numeric_cols] = media_rest_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        test_df['Q'] = test_df['VCO2'] / test_df['VO2']

         # Extract timestamps from AT and RC and convert them to timedelta
        at_time = datetime.combine(datetime.min, at_df.iloc[1]['t']) - datetime.min
        rc_time = datetime.combine(datetime.min, rc_df.iloc[1]['t']) - datetime.min

        # Convert timestamps to indices in the Test DataFrame
        at_index = np.argmin(np.abs(test_df['t'] - at_time))
        rc_index = np.argmin(np.abs(test_df['t'] - rc_time))
        is_at = [i > at_index for i in range(len(test_df['t']))]
        is_rc = [i > rc_index for i in range(len(test_df['t']))]

        # Create the label arrays for AT and RC
        test_df['label'] = 0
        for i, (at, rc) in enumerate(zip(is_at, is_rc)):
            test_df.loc[i, 'label'] = at << rc
        # test_df['label_at'] = 0
        # test_df['label_rc'] = 0
        # test_df.loc[at_index:, 'label_at'] = 1
        # test_df.loc[rc_index:, 'label_rc'] = 1
    
        
        # Normalize the Test data using Media Rest values
        for col in numeric_cols:
            if media_rest_df.iloc[0][col]:
                test_df[col] /= float(media_rest_df.iloc[0][col])

        test_df.head()

        if sum(test_df.isna().any(axis=1)) != 0:
            print("critical file path: ", file_path)

        out_data[file_path] = test_df
    return out_data


if __name__ == "__main__":
    print("Compact CPET files in a single dataset")
    print(ROOT_DIR)
    out_data = load_and_convert(Path(ROOT_DIR) / 'progetto/dati_finali/training')
    from pickle import dump
    filename = 'sequence_dataset.pkl'
    with open(Path(ROOT_DIR) / f'progetto/dati_finali/training/{filename}', 'wb') as f:
        dump(out_data, f)
    print(f"Data was written in the {filename}")