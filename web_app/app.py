from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
import pickle
import copy
import sys
sys.path.append('../src')
from data_pipeline import (files_to_dfs, clean_analyte_names,
                           calculate_cols, analyte_filter)
from data_class import Data

app = Flask(__name__)

def predict_single_batch(file_):
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
    df = read_file([file_])[0]
    cols_df = df[columns]

    # adjust names to match model
    remove_list = ['-B1a', '-B1b', ' NH4 Adduct', 'cis-', 'trans-']
    spinosad_list = ['Spinosyn A', 'Spinosyn D']
    clean_df = clean_analyte_names(cols_df, remove_list, spinosad_list)

    # update df to fit model
    cal_df = calculate_cols(clean_df)
    one_hot_df = analyte_filter(cal_df, 'All')
    fill_df = one_hot_df.fillna(0)
    mod_df = fill_df._get_numeric_data()

    return df, mod_df


def read_file(files):
    '''
    Read a single file into a pandas dataframe
    Parameters
    ----------
    file: file to open with pandas

    Return
    ------
    df_sub: DataFrame
        dataframe containing only sample information
    '''
    df_list = []
    for file_ in files:
        try:
            df = pd.read_csv(file_, skip_blank_lines=False)
            start_row = df[df.iloc[:, 0].str.startswith('Sample Name',
                                                        na=False)].index[0]
            columns = df.iloc[start_row]['Peak Name: Myclobutanil d4'].split('\t')
            df_sub = pd.DataFrame(
                [x.split('\t') for x in 
                 df.iloc[start_row+1:]['Peak Name: Myclobutanil d4']],
                columns=columns
            )
            df_sub = convert_from_str(df_sub)
            # df_sub = pd.read_csv(file_, delimiter='\t', skiprows=start_row)
            df_list.append(df_sub)
        except:
            print(f'😱{file_} does not contain "Sample Name", check the file.')
    return df_list


def line_num_for_phrase_in_file(phrase='Sample Name', filename='file.txt'):
    '''Find the line number where the phrase is located
    Parameters
    ----------
    phrase: str
        string to locate in the file
    filename: str
        file to search through
    Return
    ------
    line containing the phrase
    '''
    with open(filename, 'r') as f:
        for (i, line) in enumerate(f):
            if phrase in line:
                return i - 1
    return -1


def evaluate_batch(orig_df, mod_df, model):
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


def evaluate_controls(df):
    # Find incorrect predictions
    df['false_pos'] = np.where(
        df['predict'] == 1, np.where(df['expected'] == 0, 1, 0), 0
    )
    df['false_neg'] = np.where(
        df['predict'] == 0, np.where(df['expected'] == 1, 1, 0), 0
    )

    false_neg_df = df[df['false_neg'] == 1][['Sample Name',
                                             'Analyte Peak Name',
                                             'Analyte Peak Area (counts)']]
    false_pos_df = df[df['false_pos'] == 1][['Sample Name',
                                             'Analyte Peak Name',
                                             'Analyte Peak Area (counts)']]
    return false_neg_df, false_pos_df


def evaluate_samples(df):
    # Samples for evaluation
    df['evaluate'] = np.where(df['Sample Type'] == 'Unknown',
                              np.where(df['predict'] == 1, 1, 0), 0)
    sample_df = df[df['evaluate'] == 1][['Sample Name', 'Analyte Peak Name']]
    return sample_df


def convert_from_str(df):
    columns = ['Analyte Peak Area (counts)', 'Analyte Peak Height (cps)',
               'Analyte Retention Time (min)', 'Analyte Expected RT (min)',
               'Analyte Centroid Location (min)', 'Analyte Start Scan',
               'Analyte Start Time (min)', 'Analyte Stop Scan',
               'Analyte Stop Time (min)', 'Analyte Peak Width (min)',
               'Analyte Peak Width at 50% Height (min)',
               'Analyte Peak Asymmetry',
               'Analyte Integration Quality', 'Relative Retention Time']
    for col in columns:
        df[col] = pd.to_numeric(df[col])
    return df


@app.route('/batch_predict', methods=['GET', 'POST'])
def batch_predict():
    if request.method == 'POST':
        f = request.files['file']
        file = '../data/20200615 ws 8442.txt'
        df, mod_df = predict_single_batch(f)
        predicted_df = evaluate_batch(df, mod_df, model)
        controls_df, blanks_df = evaluate_controls(predicted_df)
        sample_df = evaluate_samples(predicted_df)
    else:
        controls_df = pd.DataFrame()
        blanks_df = pd.DataFrame()
        sample_df = pd.DataFrame()
    return render_template('batch_predict.html', controls_df=controls_df,
                           blanks_df=blanks_df, sample_df=sample_df)


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # Load model
    with open('boosted.pkl', 'rb') as f:
        model = pickle.load(f)

    app.run(host='0.0.0.0', port=8080, debug=True, threaded=True)
