import pandas as pd


class Data(object):
    '''Class to contain the different dataframes that will be created
    after calling the data pipeline. Four total dataframes, analytes one
    hot encoded or dropped and columns filtered for colinearity or not.
    Parameters
    ----------
    file: string
        file name containing data from pipeline
    analytes: string
        list of analytes included in data
        'All' for all analytes or analyte name for single analyte
    columns: list
        columns to be removed for colinearity
    '''
    def __init__(self, file, analytes, columns=None):
        self.file = file
        self.analytes = analytes
        self.columns = columns
        self.full_df = pd.read_csv(file, index_col='Unnamed: 0')
        self.limited_df = self.drop_cols(self.full_df, self.columns)
        self.no_analyte_df = self.drop_analyte_cols(self.full_df)
        self.limited_no_analyte_df = self.drop_analyte_cols(self.limited_df)

    @staticmethod
    def pop_reported(df):
        '''Pop reported column from df, return values
        Parameters
        ----------
        df: DataFrame
            dataframe containing features and target(reported)

        Return
        ------
        X: numpy array
            numpy array of all feature columns
        y: numpy array
            numpy array of targets
        '''
        df_copy = df.copy()
        y = df_copy.pop('reported').values
        X = df_copy.values
        return X, y

    @staticmethod
    def drop_cols(df, drop_cols):
        '''Drop columns from dataframe
        Parameters
        ----------
        df: DataFrame
            original df
        drop_cols: list
            list of columns to drop

        Return
        ------
        df: DataFrame
            final dataframe with columns removed
        '''
        if drop_cols is not None:
            df = df.drop(drop_cols, axis=1)
        return df

    @staticmethod
    def drop_analyte_cols(df):
        '''Drop all one hot encoded analyte columns
        Parameters
        ----------
        df: DataFrame
            dataframe containing one hot encoded analyte columns

        Return
        ------
        df_short: DataFrame
            dataframe with analyte columns dropped
        '''
        df_short = df.loc[:, ~df.columns.str.startswith('analyte_')]
        return df_short


def create_data(file, analytes):
    '''Create data class object containing data
    Parameters
    ----------
    file: string
        file name containing data from pipeline
    analytes: string
        list of analytes included in data
        'All' for all analytes or analyte name for single analyte
    Return
    ------
    data_class: Data class object
    '''
    cols_drop_list = ['Analyte Start Scan', 'Analyte Stop Scan',
                      'Analyte Centroid Location (min)',
                      'Relative Retention Time',
                      'Analyte Integration Quality',
                      'Analyte Peak Height (cps)',
                      'Analyte Peak Width at 50% Height (min)',
                      'height_ratio',
                      'area_ratio',
                      'Analyte Start Time (min)',
                      'Analyte Stop Time (min)']

    data_class = Data(file, analytes, cols_drop_list)
    return data_class


if __name__ == '__main__':

    all_df = create_data('../data/merged_df.csv', 'All')
    myclo_df = create_data('../data/merged_df_myclo.csv', 'Myclobutanil')
    print(myclo_df.analytes)
