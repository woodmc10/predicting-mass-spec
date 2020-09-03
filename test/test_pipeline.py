import unittest
import pandas as pd
import sys
sys.path.append('./src')
import data_pipeline


class TestPipeline(unittest.TestCase):
    def test_correct_sp(self):
        data = {'ComponentName': ['Marta', 'Martha', 'Marty']}
        data_correct_1 = {'ComponentName': ['Martha', 'Martha', 'Marty']}
        data_correct_2 = {'ComponentName': ['Martha', 'Martha', 'Martham']}
        df = pd.DataFrame.from_dict(data)
        df_correct_1 = pd.DataFrame.from_dict(data_correct_1)
        df_correct_2 = pd.DataFrame.from_dict(data_correct_2)
        self.assertTrue(data_pipeline.correct_sp(df, 'Marta', 'Martha')
                        .equals(df_correct_1))
        self.assertEqual(data_pipeline.correct_sp(df, 'Marty', 'Martha')['ComponentName'][2], 'Martha')
        # self.assertTrue(data_pipeline.correct_sp(df, 'Marty', 'Martha')
        #                 .equals(df_correct_2))

        # The assertEqual allows for more helpful output when the Test fails

    # def test_no_component(self):
    #     data = {'Name': ['Marta', 'Martha', 'Marty']}
    #     df = pd.DataFrame.from_dict(data)
    #     data_pipeline.correct_sp(df, 'Marta', 'Martha')
    #     # Testing what the default python error is and if it will communicate the actual problem
    #         # This one does have a useful/understandable error output

    def test_file_from_dir(self):
        dir = r'fixture/test_dir'
        file_names = ['a.txt', 'b.txt']
        self.assertCountEqual(map(lambda x: x.name, data_pipeline.files_from_dir(dir)), file_names)


    def test_files_to_dfs(self):
        files = ['data/20190506 wss 5783 PT.txt', 'data/20191220 ws 7190.txt']
        lengths = [680, 224]
        self.assertCountEqual(map(lambda x: len(x), data_pipeline.files_to_dfs(files)), lengths)


    def test_files_to_dfs_fake(self):
        files = ['fixture/fake_instrument_data.txt']
        self.assertEqual(data_pipeline.files_to_dfs(files), [])

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