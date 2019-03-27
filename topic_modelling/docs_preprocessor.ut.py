import unittest
from docs_preprocessor import DocsPreprocessor
import pandas as pd
from numpy import array
from numpy import all
import numpy

class TestDocsPreprocessor(unittest.TestCase): 
    """
    Test the DocsPreprocessor class.
    """
    
    def setUp(self):
        self.doc1 = pd.read_csv("docs/description.csv")
        self.dp = DocsPreprocessor()

    def test_tokenize_doc_type_1(self):
        result = self.dp.tokenize_doc(self.doc1)
        self.assertEqual(type(result), numpy.ndarray)
    
    def test_tokenize_doc_type_2(self):
        result = self.dp.tokenize_doc(self.doc1)
        self.assertEqual(type(result[0]), list)
    
    def test_tokenize_doc_contents_1(self):
        result = self.dp.tokenize_doc(self.doc1)
        actual_list = [
            'a', 'easy', 'way', 'to', 'use', 'android', 'sharepreference']
        self.assertEqual(result[0], actual_list)

    def test_tokenize_doc_lowercase(self):
        result = self.dp.tokenize_doc(self.doc1)
        actual_word = "webworker" # Lowercase from actual WebWorker
        self.assertEqual(result[11][1], actual_word)

    def test_remove_numbers_type(self):
        test_array = [
            'A', 'Go', 'API', 'client', 'for', 'the', 'v2', 'Docker', 
            'Registry', 'API']
        result = self.dp.remove_numbers(test_array)
        self.assertEqual(type(test_array), type(result))

    def test_remove_numbers_keep_words_with_numbers(self):
        test_array = [[
            'A', 'Go', 'API', 'client', 'for', 'the', 'v2', 'Docker', 
            'Registry', 'API']]
        result = self.dp.remove_numbers(test_array)
        self.assertEqual(test_array, result)

    def test_remove_numbers_1(self):
        test_array = [[
            'A', 'Go', 'API', 'client', 'for', 'the', 'v2', 'Docker', 
            'Registry', 'API', '2']]
        expected_result = [[
            'A', 'Go', 'API', 'client', 'for', 'the', 'v2', 'Docker', 
            'Registry', 'API']]
        result = self.dp.remove_numbers(test_array)
        self.assertEqual(result, expected_result)
    
    def test_remove_numbers_2(self):
        test_array = [[
            'A', 'Go', 'API', 'client', 'for', 'the', 'v2', 'Docker', 
            'Registry', 'API', 'two']]
        result = self.dp.remove_numbers(test_array)
        self.assertEqual(result, test_array)
        

if __name__ == '__main__':
    unittest.main()
