import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)  # To ignore all warnings that arise here to enhance clarity
import numpy as np
import unittest
from btm import BTM
from docs_preprocessor import DocsPreprocessor
import pandas as pd

class TestBTM(unittest.TestCase):
    """
    Test the BTM class.
    """
    
    def setUp(self):
        self.description_csv = pd.read_csv("docs/description.csv")
        self.description_1000_csv = pd.read_csv("docs/description_1000.csv")
        self.dp = DocsPreprocessor()
        self.description_1000 = self.dp.process(self.description_1000_csv)
        self.btm = BTM(self.description_1000)

    def test_btm(self):
        self.btm.compute_values(2, 10, 2)
        
if __name__ == '__main__':
    unittest.main()
