import pandas as pd
import numpy as np
import pickle
from data_pipeline import (files_to_dfs, clean_analyte_names,
                           calculate_cols, analyte_filter)
from data_class import Data


if __name__ == '__main__':

    file = '../data/20200615 ws 8442.txt'
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
    # Read file to df
    df = files_to_dfs([file])[0]

    # filter columns
    cols_df = df[columns]

    # adjust names to match model
    remove_list = ['-B1a', '-B1b', ' NH4 Adduct', 'cis-', 'trans-']
    spinosad_list = ['Spinosyn A', 'Spinosyn D']
    clean_df = clean_analyte_names(cols_df, remove_list, spinosad_list)

    # calculate columns to match model
    cal_df = calculate_cols(clean_df)

    # filter analytes
    one_hot_df = analyte_filter(cal_df, 'All')

    # fill na with 0
    fill_df = one_hot_df.fillna(0)

    # use only numeric columns columns
    mod_df = fill_df._get_numeric_data()

    # Create model
    with open('boosted.pkl', 'rb') as f:
        model = pickle.load(f)

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

    # Find incorrect predictions
    df['false_pos'] = np.where(
        df['predict'] == 1, np.where(df['expected'] == 0, 1, 0), 0
    )
    df['false_neg'] = np.where(
        df['predict'] == 0, np.where(df['expected'] == 1, 1, 0), 0
    )
    print(df[df['false_pos'] == 1]['Sample Name'].count())
    print(df[df['false_neg'] == 1]['Sample Name'].count())
    print(df[df['false_neg'] == 1][['Sample Name', 'Analyte Peak Name',
                                    'Analyte Peak Area (counts)']])

    # Samples for evaluation
    df['evaluate'] = np.where(df['Sample Type'] == 'Unknown',
                              np.where(df['predict'] == 1, 1, 0), 0)
    print(df[df['evaluate'] == 1][['Sample Name', 'Analyte Peak Name']])
