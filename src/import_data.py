import pandas as pd
import numpy as np
import os


def create_reported_df(analyte='All'):
    '''
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
    full_df = merge(r'../data', columns)

    # Update analyte names from instrument to match analyte names from reports
    remove_list = ['-B1a', '-B1b', ' NH4 Adduct', 'cis-', 'trans-']
    spinosad_list = ['Spinosyn A', 'Spinosyn D']
    clean_df = clean_names(full_df, remove_list, spinosad_list)

    # Create dataframe from positive reports xlsx
    pos_df = pd.read_excel('../data/pest_pos.xlsx')
    pos_df['SampleCode2'] = pos_df['SampleCode'].str[:-1]
    pos_df = correct_sp(pos_df, 'Imadacloprid_Result', 'Imidacloprid_Result')
    pos_df = correct_sp(pos_df, 'Spirotretramat_Result',
                        'Spirotetramat_Result')
    pos_df['ComponentName2'] = pos_df['ComponentName'].str.replace('_Result',
                                                                   '')

    # Merge instrument data and positive reports
    reported_full_df = merge_reports(clean_df, pos_df)

    if analyte == 'All':
        # One-hot-encode analytes
        reported_full_df = pd.get_dummies(reported_full_df,
                                          columns=['analyte'],
                                          drop_first=True)
        # drop_first drops Abamectin
    else:
        # Filter on analyte name
        reported_full_df = reported_full_df[reported_full_df['analyte'] ==
                                            analyte]

    return reported_full_df


def merge(directory, cols):
    count = 0
    for entry in os.scandir(directory):
        if entry.path.endswith(".txt") and entry.is_file():
            df = import_single(entry)
            reduced_df = reduce_df(df, cols)
        if count == 0:
            merged_df = reduced_df
        else:
            merged_df = merged_df.append(reduced_df)
        count += 1
    return merged_df


def import_single(file):
    df = pd.read_csv(file, skip_blank_lines=False)
    start_row = df[df.iloc[:, 0].str.startswith('Sample Name',
                                                na=False)].index[0]
    df_sub = pd.read_csv(file, delimiter='\t', skiprows=start_row)
    return df_sub


def reduce_df(df, cols):
    df_cols = df[cols]
    df_rows_dropped = df_cols[df_cols['Analyte Retention Time (min)'] != 0]
    sample_df = df_rows_dropped[df_rows_dropped['Sample Type'] == 'Unknown']
    return sample_df


def clean_names(df, remove_list, spinosad_list):
    # df['sample_name'] = df['Sample Name'].str[-11:]
    # regex_pattern = (\d{8}-\d{2})
    # pattern = '(\d{8}-\d{2})'
    # prog = re.compile(pattern)
    df['sample_name'] = df['Sample Name'].str.extract(r'(\d{8}-\d{2})')
    df.dropna(subset=['sample_name'], inplace=True)
    df['analyte'] = df['Analyte Peak Name'].str[:-2]
    df = df[df['Relative Retention Time'] != 0]
    for elem in remove_list:
        df['analyte'] = df['analyte'].str.replace(elem, '')
    for elem in spinosad_list:
        df['analyte'] = df['analyte'].str.replace(elem, 'Spinosad')
    return df


def correct_sp(df, incorrect, correct):
    df['ComponentName'] = df['ComponentName'].str.replace(incorrect, correct)
    return df


def merge_reports(full_df, reports_df):
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

    reported_df = create_reported_df('Myclobutanil')

    print('Count:', reported_df['reported'].count())
    print('Reported:', reported_df['reported'].sum())

    # Calculate new columns and create numeric columns from object columns
    add_cols_df = calculate_cols(reported_df)

    # Drop all non-numeric columns
    numeric_df = add_cols_df._get_numeric_data()

    # NaNs are from #DIV/0 errors in Analyte Slope of Baseline (%/min)
    zero_na_df = numeric_df.fillna(0)

    zero_na_df.to_csv('../data/merged_df_myclo.csv')