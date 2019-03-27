import unittest
from docs_preprocessor import DocsPreprocessor
import pandas as pd
import numpy

class TestDocsPreprocessor(unittest.TestCase): 
    """
    Test the DocsPreprocessor class.
    """
    
    def setUp(self):
        self.doc1 = pd.read_csv("docs/description.csv")
        self.dp = DocsPreprocessor()

    def test_tokenize_doc_type_1(self):
        test_result = self.dp.tokenize_doc(self.doc1)
        self.assertEqual(type(test_result), numpy.ndarray)
    
    def test_tokenize_doc_type_2(self):
        test_result = self.dp.tokenize_doc(self.doc1)
        self.assertEqual(type(test_result[0]), list)

if __name__ == '__main__':
    unittest.main()
