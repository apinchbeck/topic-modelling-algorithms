import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)  # To ignore all warnings that arise here to enhance clarity
import numpy as np
import unittest
from nmf import NMF
from docs_preprocessor import DocsPreprocessor
from modelling import Modelling
import pandas as pd

class TestModelling(unittest.TestCase):
    """
    Test the Modelling class.
    """
    def setUp(self):
        self.description = pd.read_csv("docs/description.csv")
        self.description_1000 = pd.read_csv("docs/description_1000.csv")
        self.mdl = Modelling(self.description_1000)
    
    # def test_run_lda(self):
    #     self.mdl.run_lda(10, 100, 10)
    
    def test_run_nmf(self):
        self.mdl.run_nmf(10, 100, 10, 20)

if __name__ == '__main__':
    unittest.main()