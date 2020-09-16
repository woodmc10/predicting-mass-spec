import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from data_pipeline import (files_from_dir, files_to_dfs, clean_analyte_names,
                           calculate_cols, analyte_filter)
from data_class import Data


def predict_batches(directory):
    columns = ['Sample Name', 'Sample Type', 'Analyte Peak Name',
               'Analyte Peak Area (counts)', 'Analyte Peak Height (cps)',
               'Analyte Retention Time (min)', 'Analyte Expected RT (min)',
               'Analyte Centroid Location (min)', 'Analyte Start Scan',
               'Analyte Start Time (min)', 'Analyte Stop Scan',
               'Analyte Stop Time (min)', 'Analyte Peak Width (min)',
               'Area Ratio', 'Height Ratio',
               'Analyte Peak Width at 50% Height (min)',
               'Analyte Slope of Baseline (%/min)', 'Analyte Peak Asymmetry',
               'Analyte Integration Quality', 'Relative Retention Time']

    # Create df for model
    # Read files to dfs
    files = files_from_dir(directory)
    dfs = files_to_dfs(files)
    merged_df = pd.concat(dfs)
    cols_df = merged_df[columns]

    # adjust names to match model
    remove_list = ['-B1a', '-B1b', ' NH4 Adduct', 'cis-', 'trans-']
    spinosad_list = ['Spinosyn A', 'Spinosyn D']
    clean_df = clean_analyte_names(cols_df, remove_list, spinosad_list)

    # update df to fit model
    cal_df = calculate_cols(clean_df)
    one_hot_df = analyte_filter(cal_df, 'All')
    fill_df = one_hot_df.fillna(0)
    mod_df = fill_df._get_numeric_data()

    return merged_df, mod_df


def predict_batch(orig_df, mod_df, model):
    df = orig_df.copy()
    # Predict probabilities
    probs = model.predict_proba(mod_df)

    # Add predictions to df
    df['probs'] = probs[:, 1]
    df['predict'] = np.where(df['Analyte Retention Time (min)'] == 0, 0,
                             np.where(df['probs'] > 0.4, 1, 0))

    # Add expected results
    df['expected'] = np.where(
        df['Sample Type'] == 'Quality Control', 1,
        np.where(df['Sample Type'] == 'Standard', 1,
                 np.where(df['Sample Type'] == 'Blank', 0,
                          np.where(df['Sample Type'] == 'Solvent', 0, None)))
    )
    return df


def predict_controls(df):
    # Find incorrect predictions
    df['false_pos'] = np.where(
        df['predict'] == 1, np.where(df['expected'] == 0, 1, 0), 0
    )
    df['false_neg'] = np.where(
        df['predict'] == 0, np.where(df['expected'] == 1, 1, 0), 0
    )
    return df


def predict_samples(df):
    # Samples for evaluation
    df['sample_pred'] = np.where(df['Sample Type'] == 'Unknown',
                                 np.where(df['predict'] == 1, 1, 0), 0)
    return df


if __name__ == '__main__':

    directory = '../data'

    # Create model
    with open('boosted.pkl', 'rb') as f:
        model = pickle.load(f)

    full_df, mod_df = predict_batches(directory)
    df_preds = predict_batch(full_df, mod_df, model)
    df_controls= predict_controls(df_preds)
    df_final = predict_samples(df_controls)
    df_final['Acquisition Date'] = pd.to_datetime(
        df_final['Acquisition Date']
    )
    df_final['false_neg_adj'] = np.where(
        df_final['false_neg'] == 1,
        np.where(df_final['Analyte Peak Area (counts)'] > 0, 1, 0),
        0
    )
    df_final['profit'] = (df_final['sample_pred'] * 0.5
                          - df_final['false_neg_adj'] * 100)
    df_final.set_index('Acquisition Date', inplace=True)
    df_final.sort_index(inplace=True)
    df_final['profit_total'] = df_final['profit'].cumsum()
    df_final['profit_total'].plot(color='green', figsize=(10, 5))
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Profits ($)', fontsize=14)
    plt.title('Profits Over Time', fontsize=16)
    plt.savefig('../images/profit_time.png')