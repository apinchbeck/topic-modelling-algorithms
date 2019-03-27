import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)  # To ignore all warnings that arise here to enhance clarity
import numpy as np
import unittest
from nmf import NMF
from docs_preprocessor import DocsPreprocessor
import pandas as pd

class TestNMF(unittest.TestCase):
    """
    Test the NMF class.
    """
    
    def setUp(self):
        self.description_csv = pd.read_csv("docs/description.csv")
        self.description_1000_csv = pd.read_csv("docs/description_1000.csv")
        self.dp = DocsPreprocessor()
        self.description_1000 = self.dp.process(self.description_1000_csv)
        self.nmf = NMF(self.description_1000)

    def test_type(self):
        self.assertEqual(type(self.nmf.docs), list)
        
    """ def test_vectorize(self):
        vect, terms = self.nmf.vectorize()
        self.assertTrue(len(terms) == 2381)
        self.assertEqual((vect.shape[0], vect.shape[1]), (1000, 2381)) """

    """ def test_create_model(self):
        self.nmf.create_model(10) """

    """ def test_run_topic_models(self):
        self.nmf.run_topic_models(10, 30, 10) """
    
    """ def test_create_word_embedding_model(self):
        w_model = self.nmf.create_word_embedding_model() """

    def test_process_models(self):
        self.nmf.process_models(10, 30, 10, 20)

if __name__ == '__main__':
    unittest.main()
