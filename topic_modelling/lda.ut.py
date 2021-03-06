import unittest
from lda import LDA
from docs_preprocessor import DocsPreprocessor
import pandas as pd

class TestLDA(unittest.TestCase): 
    """
    Test the LDA class.
    """

    def setUp(self):
        self.description_csv = pd.read_csv("docs/description.csv")
        self.description_1000_csv = pd.read_csv("docs/description_1000.csv")
        self.dp = DocsPreprocessor()
        self.description_1000 = self.dp.process(self.description_1000_csv)
        self.lda = LDA(self.description_1000)

    def test_1(self):
        k_values, coherence_values, topic_list = self.lda.compute_coherence_values(5, 20, 5)
        

if __name__ == '__main__':
    unittest.main()
