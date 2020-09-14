import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler



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
    
    def filter_cols(self, filter_type=None):
        '''Filter train_df and test_df to include only columns of interest
        Parameters
        ----------
        filter_type: str
            options: limited, no_analyte, limited_no_analyte
        '''
        if filter_type == 'limited':
            filtered_df = self.drop_cols(self.full_df, self.columns)
        elif filter_type == 'no_analyte':
            filtered_df = self.drop_analyte_cols(self.full_df)
        elif filter_type == 'limited_no_analyte':
            filtered_df = self.drop_analyte_cols(self.drop_cols(self.full_df,
                                                                self.columns))
        elif filter_type is None:
            filtered_df = self.full_df
        else:
            print('Warning: incorrect filter_type passed to filter_cols')
            filtered_df = None
        return filtered_df
    
    def train_test_split(self, test_size=0.33, filter_type=None):
        '''Train test split the df for testing
        Parameters
        ----------
        train_size: float
            percent of dataframe for testing data
        filter_type: str ('limited', 'no_analyte', 'limited_no_analyte')
            description of how to filter data
        '''
        df = self.filter_cols(filter_type)
        X, y = self.pop_reported(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
            )
        self.test_df = pd.merge(X_test, y_test, how='left',
                                left_index=True, right_index=True)
        self.train_df = pd.merge(X_train, y_train, how='left',
                                 left_index=True, right_index=True)
    
    def over_sampling(self, df):
        '''Over sample the positive class 
        Parameters
        ----------
        df: pandas DataFrame
            dataframe of samples for over sampling
        Return
        ------
        X_ros: 
            random over sampling of the features
        y_ros:
            random over sampling of the targets
        '''
        ros = RandomOverSampler(random_state=42)
        X, y = self.pop_reported(df)
        X_ros, y_ros = ros.fit_sample(X, y)
        return X_ros, y_ros

    def under_sampling(self, df):
        '''Under sample the positive class 
        Parameters
        ----------
        df: pandas DataFrame
            dataframe of samples for over sampling
        Return
        ------
        X_rus: 
            random under sampling of the features
        y_rus:
            random under sampling of the targets
        '''
        rus = RandomUnderSampler(random_state=42)
        X, y = self.pop_reported(df)
        X_rus, y_rus = rus.fit_sample(X, y)
        return X_rus, y_rus

    def smote_sampling(self, df):
        '''SMOTE over sample the positive class 
        Parameters
        ----------
        df: pandas DataFrame
            dataframe of samples for over sampling
        Return
        ------
        X_smote: 
            SMOTE over sampling of the features
        y_smote:
            SMOTE over sampling of the targets
        '''
        smote = SMOTE(random_state=42)
        X, y = self.pop_reported(df)
        X_smote, y_smote = smote.fit_sample(X, y)
        return X_smote, y_smote

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
        y = df_copy.pop('reported')
        X = df_copy
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
    # check class creation
    all_df = create_data('../data/merged_df.csv', 'All')
    myclo_df = create_data('../data/merged_df_myclo.csv', 'Myclobutanil')
    print(myclo_df.analytes)
