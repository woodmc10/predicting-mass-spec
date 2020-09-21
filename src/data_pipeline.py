import pandas as pd
import numpy as np
import os
import functools
import time


def create_reported_df(path=r'../data', pos_file='../data/pest_pos.xlsx'):
    '''
    Full pipeline calling all other functions to create the final dataframe
    Parameters
    ----------
    anlayte: str
        'All' or a name of an analyte in the dataframe.
        'All' will one hot encode the analyte
        column, an analyte name will filter the dataframe
        to include just that analyte.

    Return
    ------
    df: pandas DataFrame
        DataFrame containing all features before feature engineering
    '''
    # Columns to keep (include columns about samples and analytes)
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

    # Merge all files in data folder
    full_df = merge(path, columns)

    # Update analyte names from instrument to match analyte names from reports
    sample_df = clean_sample_names(full_df)
    remove_list = ['-B1a', '-B1b', ' NH4 Adduct', 'cis-', 'trans-']
    spinosad_list = ['Spinosyn A', 'Spinosyn D']
    clean_df = clean_analyte_names(sample_df, remove_list, spinosad_list)

    # Create dataframe from positive reports xlsx
    pos_df = pd.read_excel(pos_file)
    pos_df['SampleCode2'] = pos_df['SampleCode'].str[:-1]
    pos_df = correct_sp(pos_df, 'Imadacloprid_Result', 'Imidacloprid_Result')
    pos_df = correct_sp(pos_df, 'Spirotretramat_Result',
                        'Spirotetramat_Result')
    pos_df['ComponentName2'] = pos_df['ComponentName'].str.replace('_Result',
                                                                   '')

    # Merge instrument data and positive reports
    reported_full_df = merge_reports(clean_df, pos_df)

    return reported_full_df


def analyte_filter(df, analyte='All'):
    '''One hot encode the analyte column for all analytes or one analyte
    Parameters
    ----------
    df: pandas DataFrame
        dataframe including an 'analyte' column
    analyte: str
        name of analyte to keep or 'All' for all analytes
    Returns
    -------
    one_hot_df: pandas DataFrame
        dataframe with one hot encode columns for analytes
    '''
    if analyte == 'All':
        # One-hot-encode analytes
        one_hot_df = pd.get_dummies(df, columns=['analyte'], drop_first=False)
        # drop_first drops Abamectin
    else:
        # Filter on analyte name
        filtered_df = df[df['analyte'] == analyte]
        one_hot_df = pd.get_dummies(filtered_df, columns=['analyte'])
    return one_hot_df


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        return value
    return wrapper_timer


@timer
def merge(directory, cols):
    '''Merge all csv files in the directory
    Parameters
    ----------
    directory: re string
        folder containing files to convert into dataframe
    cols: list
        list of columns to include in dataframes

    Returns
    -------
    merged_df: DataFrame
        dataframe from all txt files filtered to only contain columns
        in the cols list
    '''
    files = files_from_dir(directory)
    dfs = files_to_dfs(files)
    reduced_dfs = reduce_df(dfs, cols)
    merged_df = pd.concat(reduced_dfs)
    return merged_df


def files_from_dir(directory):
    '''Loop through all files in a directory and create a list of
    .txt files
    Parameter
    ---------
    directory: re string
        folder containing .txt files
    Return
    ------
    list of .txt files in directory
    '''
    return [entry for entry in os.scandir(directory)
            if entry.name.endswith('.txt') and entry.is_file()]


def files_to_dfs(files):
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
        start_row = line_num_for_phrase_in_file('Sample Name', file_)
        if start_row >= 0:
            df_sub = pd.read_csv(file_, delimiter='\t', skiprows=start_row)
            df_list.append(df_sub)
        else:
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


def reduce_df(dfs, cols):
    ''' drop unnecessary columns from dataframe, drop samples that do not
    have integrated peaks (RT = 0) and all controls (sample type <> unknown)
    Parameters:
    ----------
    df: DataFrame
        original dataframe
    cols: list
        columns to keep in dataframe

    Return
    ------
    sample_df: DataFrame
        filtered dataframe
    '''
    df_list = []
    for df in dfs:
        df_cols = df[cols]
        df_rrt = df_cols[df_cols['Relative Retention Time'] != 0]
        df_rows_dropped = df_rrt[df_rrt['Analyte Retention Time (min)'] != 0]
        sample_df = df_rows_dropped[df_rows_dropped['Sample Type']
                                    == 'Unknown']
        df_list.append(sample_df)
    return df_list


def clean_sample_names(df):
    '''Adjust sample names to match reported list and drop names that
    do not fit the sample name format
    Parameters
    ----------
    df: DataFrame
        merged datafram of all samples
    Return
    ------
    df: DataFrame
        dataframe with sample names cleaned
    '''
    df['sample_name'] = df['Sample Name'].str.extract(r'(\d{8}-\d{2})')
    df.dropna(subset=['sample_name'], inplace=True)
    return df


def clean_analyte_names(df, remove_list, spinosad_list):
    ''' Adjust analyte name to match reported list from lab
    Parameters
    ----------
    df: DataFrame
        merged dataframe of all samples
    remove_list: list
        list of string to remove from analyte names
    spinosad_list: list
        list of strings to replace with 'Spinosad'
        necessary be Spinosad is composed of Spinosyn A and Spinosyn D
    Return
    ------
    df: DataFrame
        dataframe with corrected names
    '''
    df['analyte'] = df['Analyte Peak Name'].str[:-2]
    for elem in remove_list:
        df['analyte'] = df['analyte'].str.replace(elem, '')
    for elem in spinosad_list:
        df['analyte'] = df['analyte'].str.replace(elem, 'Spinosad')
    return df


def correct_sp(df, incorrect, correct):
    '''Correct spelling of names incorrectly typed into report list
    Parameters
    ----------
    df: DataFrame
        dataframe of reported analyte
    incorrect: string
        incorrectly spelled analyte name
    correct: string
        correctly spelled analyte name

    Return
    ------
    df: DataFrame
        dataframe with all analyte names spelled correctly
    '''
    df['ComponentName'] = df['ComponentName'].str.replace(incorrect, correct)
    return df


def merge_reports(full_df, reports_df):
    '''Merge instrument report dataframe with reported sample dataframe
    Parameters
    ----------
    full_df: DataFrame
        dataframe containing instrument data
    reports_df: DataFrame
        dataframe containing reported samples

    Return
    ------
    df: DataFrame
        dataframe where reported samples are listed
    '''
    df = full_df.merge(reports_df[['Numeric Result', 'SampleCode2',
                                   'ComponentName2']],
                       how='left', left_on=['sample_name', 'analyte'],
                       right_on=['SampleCode2', 'ComponentName2'])
    df.drop(['SampleCode2', 'ComponentName2'], axis=1, inplace=True)
    df.fillna(0, inplace=True)
    df['reported'] = np.where(df['Numeric Result'] == 0, 0, 1)
    df.drop(['Numeric Result'], axis=1, inplace=True)
    return df


def calculate_cols(df):
    '''Make numeric calculations on columns for rt_diff and convert object
    columns to floats
    Parameters
    ----------
    df: DataFrame
        original dataframe

    Returns
    -------
    df: DataFrame
        updated dataframe
    '''
    df['rt_diff'] = abs(df['Analyte Retention Time (min)']
                        - df['Analyte Expected RT (min)'])
    df.drop(['Analyte Retention Time (min)', 'Analyte Expected RT (min)'],
            axis=1, inplace=True)
    df['area_ratio'] = pd.to_numeric(df['Area Ratio'], errors='coerce')
    df['height_ratio'] = pd.to_numeric(df['Height Ratio'], errors='coerce')
    df['baseline'] = pd.to_numeric(df['Analyte Slope of Baseline (%/min)'],
                                   errors='coerce')
    return df


if __name__ == '__main__':

    reported_df = create_reported_df()
    reported_df = analyte_filter(reported_df, 'All')

    print('Count:', reported_df['reported'].count())
    print('Reported:', reported_df['reported'].sum())

    # Calculate new columns and create numeric columns from object columns
    add_cols_df = calculate_cols(reported_df)

    # Drop all non-numeric columns
    numeric_df = add_cols_df._get_numeric_data()

    # NaNs are from #DIV/0 errors in Analyte Slope of Baseline (%/min)
    zero_na_df = numeric_df.fillna(0)

    zero_na_df.to_csv('../data/merged_df_test.csv')
