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
        self.description_csv = pd.read_csv("docs/description.csv")
        self.description_1000_csv = pd.read_csv("docs/description_1000.csv")
        self.mdl = Modelling(self.description_csv)
    
    def test_run_lda(self):
        self.mdl.run_algorithms(5, 20, 5, 20)

if __name__ == '__main__':
    unittest.main()