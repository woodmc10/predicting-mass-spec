import pandas as pd


class Data(object):
    
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
        df_copy = df.copy()
        y = df_copy.pop('reported').values
        X = df_copy.values
        return X, y

    @staticmethod
    def drop_cols(df, drop_cols):
        if drop_cols is not None:
            df = df.drop(drop_cols, axis=1)
        return df

    @staticmethod
    def drop_analyte_cols(df):
        df_short = df.loc[:, ~df.columns.str.startswith('analyte_')]
        return df_short


if __name__ == '__main__':

    cols_drop_list = ['Analyte Start Scan', 'Analyte Stop Scan',
                        'Analyte Centroid Location (min)',
                        'Relative Retention Time',
                        'Analyte Integration Quality',
                        # 'Analyte Peak Area (counts)',
                        'Analyte Peak Height (cps)',
                        'Analyte Peak Width (min)',
                        'Analyte Peak Width at 50% Height (min)',
                        'height_ratio',
                        'area_ratio',
                        'Analyte Start Time (min)',
                        'Analyte Stop Time (min)']

    all_df = Data('../data/merged_df.csv', 'All', cols_drop_list)
    myclo_df = Data('../data/merged_df_myclo.csv', 'Myclobutanil',
                    cols_drop_list)
    print(myclo_df.analytes)