import unittest
import pandas as pd
import sys
sys.path.append('./src')
import data_pipeline


class TestPipeline(unittest.TestCase):
    def test_correct_sp(self):
        data = {'ComponentName': ['Marta', 'Martha', 'Marty']}
        data_correct_1 = {'ComponentName': ['Martha', 'Martha', 'Marty']}
        data_correct_2 = {'ComponentName': ['Marta', 'Martha', 'Martha']}
        df = pd.DataFrame.from_dict(data)
        df_correct_1 = pd.DataFrame.from_dict(data_correct_1)
        df_correct_2 = pd.DataFrame.from_dict(data_correct_2)
        self.assertEqual(data_pipeline.correct_sp(df, 'Marta', 'Martha'), df_correct_1)
        self.assertEqual(data_pipeline.correct_sp(df, 'Marty', 'Martha'), df_correct_2)
        # TODO: currently returns ValueError: The truth value of a DataFrame is ambiguous.


if __name__ == '__main__':
    unittest.main()