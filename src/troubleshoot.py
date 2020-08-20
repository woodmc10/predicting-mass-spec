import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os

def import_single(file):
    df = pd.read_csv(file, skip_blank_lines=False)
    start_row = df[df.iloc[:,0].str.startswith('Sample Name', na=False)].index[0]
    df_sub = pd.read_csv(file, delimiter='\t', skiprows= start_row)
    return df_sub

def reduce(df, cols):
    df_cols_dropped = df[cols]
    # df_cols_dropped = df_cols[df_cols['Analyte Retention Time (min)'] != 0]
    sample_df = df_cols_dropped[df_cols_dropped['Sample Type'] == 'Unknown']
    return sample_df

def merge(directory, cols):
    count = 0
    for entry in os.scandir(directory):
        if entry.path.endswith(".txt") and entry.is_file():
            df = import_single(entry)
            reduced_df = reduce(df, cols)
        if count == 0:
            merged_df = reduced_df
        else:
            merged_df = merged_df.append(reduced_df)
        count += 1
    return merged_df

def clean_names(df, remove_list, spinosad_list):
    df['sample_name'] = df['Sample Name'].str[-11:]
    df['analyte'] = df['Analyte Peak Name'].str[:-2]
    for elem in remove_list:
        df['analyte'] = df['analyte'].str.replace(elem, '')
    for elem in spinosad_list:
        df['analyte'] = df['analyte'].str.replace(elem, 'Spinosad')
    return df


def merge_reports(full_df, reports_df):
    reported_full_df = full_df.merge(reports_df[['Numeric Result','SampleCode2',
                                                 'ComponentName2']],
                                     how='left', left_on=['sample_name', 'analyte'],
                                     right_on=['SampleCode2', 'ComponentName2'])
    reported_full_df.drop(['SampleCode2', 'ComponentName2'], axis=1, inplace=True)
    reported_full_df.fillna(0, inplace=True)
    reported_full_df['reported'] = np.where(reported_full_df['Numeric Result'] == 0, 0, 1)
    return reported_full_df

def calculate_cols(df):
    df['rt_diff'] = abs(df['Analyte Retention Time (min)'] - df['Analyte Expected RT (min)'])
    df['time_diff'] = abs(df['Analyte Stop Scan'] - df['Analyte Start Scan'])
    df['area_ratio'] = pd.to_numeric(df['Area Ratio'], errors='coerce')
    df['height_ratio'] = pd.to_numeric(df['Height Ratio'], errors='coerce')
    df['baseline'] = pd.to_numeric(df['Analyte Slope of Baseline (%/min)'], errors='coerce')
    return df

def feature_engineer(df):
    #drop name columns and area/height (used to determine positives)
#     important_cols_df = df.drop(['Sample Name', 'Sample Type', 'Analyte Peak Name'], axis=1)
    important_cols_df = df.drop(['Analyte Peak Area (counts)', 'Analyte Peak Height (cps)'],
                                 axis=1)
    #create an rt_diff and time_diff column then drop columns used to create
    important_cols_df.drop(['Analyte Retention Time (min)', 'Analyte Expected RT (min)'],
                           axis=1, inplace=True)
    
    important_cols_df.drop(['Analyte Start Time (min)', 'Analyte Stop Time (min)'],
                           axis=1, inplace=True)
    #drop start and stop scan columns (similar to start and stop time columns)
    important_cols_df.drop(['Analyte Start Scan', 'Analyte Stop Scan'], axis=1, inplace=True)
    
    #drop columns for final model df
#     reported_df = important_cols_df.drop(['sample_name', 'analyte', 'Numeric Result',
    reported_df = important_cols_df.drop(['Analyte Centroid Location (min)', 
                                     'Relative Retention Time'], axis=1)
    #convert object type columns to numbers
    
#     reported_df.drop(['Area Ratio', 'Height Ratio', 'Analyte Slope of Baseline (%/min)'],
#                     axis = 1, inplace=True)
    return reported_df

def variance_factor(df):
    X = df.drop('reported', axis=1)
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif["features"] = X.columns
    return vif


if __name__ == '__main__':
    columns = ['Sample Name', 'Sample Type', 'Analyte Peak Name', 'Analyte Peak Area (counts)',
            'Analyte Peak Height (cps)',
            'Analyte Retention Time (min)', 'Analyte Expected RT (min)',
            'Analyte Centroid Location (min)', 'Analyte Start Scan', 'Analyte Start Time (min)',
            'Analyte Stop Scan', 'Analyte Stop Time (min)', 'Analyte Peak Width (min)',
            'Area Ratio', 'Height Ratio', 'Analyte Peak Width at 50% Height (min)',
            'Analyte Slope of Baseline (%/min)', 'Analyte Peak Asymmetry',
            'Analyte Integration Quality', 'Relative Retention Time']

    full_df = merge(r'../data', columns)
    print(full_df.head().to_markdown())

    remove_list = ['-B1a', '-B1b', ' NH4 Adduct', 'cis-', 'trans-']
    spinosad_list = ['Spinosyn A', 'Spinosyn D']
    clean_df = clean_names(full_df, remove_list, spinosad_list)
    clean_df.head()

    pos_df = pd.read_excel('../data/pest_pos.xlsx')
    pos_df['SampleCode2'] = pos_df['SampleCode'].str[:-1]
    pos_df['ComponentName2'] = pos_df['ComponentName'].str.replace('_Result', '')
    pos_df.info()

    reported_full_df = merge_reports(full_df, pos_df)

    add_cols_df = calculate_cols(reported_full_df)

    numeric_df = add_cols_df._get_numeric_data()

    drop_na_df = numeric_df.dropna()

    zero_na_df = numeric_df.fillna(0)

    pd.plotting.scatter_matrix(add_cols_df, figsize=(12,12))

    add_cols_df.plot.scatter(x='Relative Retention Time', y='Analyte Centroid Location (min)')
    plt.savefig('../images/eda_scatter_1')

    # add_cols_df.plot.scatter(x='time_diff', y='Analyte Peak Width (min)')

    numeric_df['baseline'].hist(bins = 100)

    print(variance_factor(zero_na_df).to_markdown())
    
    df_1 = feature_engineer(zero_na_df)

    print(variance_factor(df_1).to_markdown())

    df_2 = df_1.drop(['Analyte Peak Width (min)', 'Numeric Result', 'area_ratio',
                      'time_diff', 'Analyte Peak Width at 50% Height (min)'], axis=1)

    print(variance_factor(df_2).to_markdown())

    # df_2.plot.scatter(x='Analyte Peak Asymmetry', y='Analyte Integration Quality')

    # plt.show()
