import unittest
import pandas as pd
import sys
sys.path.append('./src')
import data_pipeline


class TestPipeline(unittest.TestCase):
    def test_correct_sp(self):
        data = {'ComponentName': ['Marta', 'Martha', 'Marty']}
        data_correct_1 = {'ComponentName': ['Martha', 'Martha', 'Marty']}
        data_correct_2 = {'ComponentName': ['Martha', 'Martha', 'Martha']}
        df = pd.DataFrame.from_dict(data)
        df_correct_1 = pd.DataFrame.from_dict(data_correct_1)
        df_correct_2 = pd.DataFrame.from_dict(data_correct_2)
        self.assertTrue(data_pipeline.correct_sp(df, 'Marta', 'Martha').equals(df_correct_1))
        self.assertTrue(data_pipeline.correct_sp(df, 'Marty', 'Martha').equals(df_correct_2))


if __name__ == '__main__':
    unittest.main()