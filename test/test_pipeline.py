import unittest
import pandas as pd
import sys
sys.path.append('./src')
import data_pipeline
import great_expectations as ge


class TestPipeline(unittest.TestCase):
    def test_correct_sp(self):
        data = {'ComponentName': ['Marta', 'Martha', 'Marty']}
        data_correct_1 = {'ComponentName': ['Martha', 'Martha', 'Marty']}
        df = pd.DataFrame.from_dict(data)
        df_correct_1 = pd.DataFrame.from_dict(data_correct_1)
        self.assertTrue(data_pipeline.correct_sp(df, 'Marta', 'Martha')
                        .equals(df_correct_1))
        self.assertEqual(data_pipeline.correct_sp(df, 'Marty', 'Martha')
                         ['ComponentName'][2], 'Martha')
        # The assertEqual allows for more informative output if the Test fails

    def test_file_from_dir(self):
        dir = r'fixture/test_dir'
        file_names = ['a.txt', 'b.txt']
        self.assertCountEqual(map(lambda x: x.name,
                              data_pipeline.files_from_dir(dir)), file_names)

    def test_files_to_dfs(self):
        files = ['data/20190506 wss 5783 PT.txt', 'data/20191220 ws 7190.txt']
        lengths = [680, 224]
        self.assertCountEqual(map(lambda x: len(x),
                              data_pipeline.files_to_dfs(files)), lengths)

    def test_files_to_dfs_fake(self):
        files = ['fixture/fake_instrument_data.txt']
        self.assertEqual(data_pipeline.files_to_dfs(files), [])

    def test_reduce_df(self):
        data = {'Analyte Retention Time (min)': [0, 1, 2],
                'Sample Type': ['Unknown', 'Unknown', 'Standard'],
                'Relative Retention Time': [1, 0, 2],
                'Drop Me': [1, 2, 3]}
        df = pd.DataFrame.from_dict(data)
        self.assertEqual(
            data_pipeline.reduce_df([df],['Analyte Retention Time (min)',
                                          'Relative Retention Time',
                                          'Sample Type'])[0].shape,
            (0, 3)
        )

    def test_clean_names(self):
        data = {'Sample Name': ['20090501-10ABC', 'RM20090501-10',
                                '12320090501-10', '20090501-10123'],
                'Analyte Peak Name': ['Too Clever-1', 'Not Clever Enough-2',
                                      'Very Clever-1', 'Just Odd-2']}
        df = pd.DataFrame.from_dict(data)
        remove_list = ['Too ', 'Not ', ' Enough', 'Very ', 'Just ']
        spinosad_list = ['Odd']
        self.assertCountEqual(
            data_pipeline.clean_names(df, remove_list, spinosad_list)
            ['sample_name'].unique(), ['20090501-10']
        )
        self.assertEqual(
            data_pipeline.clean_names(df, remove_list, spinosad_list).shape,
            (4, 4)
        )
        self.assertCountEqual(
            data_pipeline.clean_names(df, remove_list, spinosad_list)
            ['analyte'].unique(), ['Clever', 'Spinosad']
        )

    def test_merge_reports(self):
        full_data = {'sample_name': ['Sample1', 'Sample2'],
                     'analyte': ['Analyte1', 'Analyte1']}
        full_df = pd.DataFrame.from_dict(full_data)
        report_data = {'Numeric Result': [2],
                       'SampleCode2': ['Sample1'],
                       'ComponentName2': ['Analyte1']
                       }
        report_df = pd.DataFrame.from_dict(report_data)
        self.assertEqual(
            data_pipeline.merge_reports(full_df, report_df).shape,
            (2, 3)
        )
        self.assertEqual(
            data_pipeline.merge_reports(full_df, report_df)['reported'][0],
            1
        )
        self.assertEqual(
            data_pipeline.merge_reports(full_df, report_df)['reported'][1],
            0
        )

    def test_expectations(self):
        df = data_pipeline.create_reported_df('All', r'data/',
                                              'data/pest_pos.xlsx')
        ge_df = ge.from_pandas(df)
        test1 = ge_df.expect_column_values_to_not_be_null('sample_name')
        self.assertEqual([], test1.result['partial_unexpected_list'])

        test2 = ge_df.expect_column_values_to_be_in_set('reported', [0,1])
        self.assertEqual([], test2.result['partial_unexpected_list'])

        test3 = ge_df.expect_column_values_to_be_in_set('reported', [1],
                                                        mostly=0.01)
        self.assertLessEqual(0.99, test3.result['unexpected_percent'])

        test4 = ge_df.expect_column_values_to_match_regex(
            column='sample_name', regex=r'[0-9]{8}\-[0-9]{2}'
        )
        self.assertEqual([], test4.result['partial_unexpected_list'])

        test5 = ge_df.expect_column_values_to_be_between(
            'Analyte Retention Time (min)', 0, 14
        )
        self.assertEqual([], test5.result['partial_unexpected_list'])
        '''
        Things to check
            - √at least 1% reported
            - analyte names - would need to run this before one_hot
            - √sample names
            - no nans in df - does this by column (which columns to check?)
            - all analytes present - would need list of all columns
            - √retention times < 14 (check with Nick about method run time)

        '''

if __name__ == '__main__':
    unittest.main()

    #S - single responsibility (do one thing and do it well)
    #O
    #L
    #I
    #D

    # Autosave in VS Code (turn on?)

    # terminal ctrl r (reverse search)

    # ctrl cmd space = opens emoji keyboard
        # adding emojis to error messages helps to filter for relevant info