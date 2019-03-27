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
        
        # self.test_array1
        # ['a', 'easy', 'way', 'to', 'use', 'android', 'sharepreference']
        self.test_array1 = [
            'a', 'easy', 'way', 'to', 'use', 'android', 'sharepreference']
        # self.test_array2
        # [['A', 'Go', 'API', 'client', 'for', 'the', 'v2', 'Docker', 
        #    'Registry', 'API']]
        self.test_array2 = [[
            'A', 'Go', 'API', 'client', 'for', 'the', 'v2', 'Docker', 
            'Registry', 'API']]
        # self.test_array3
        # [['A', 'Go', 'API', 'client', 'for', 'the', 'v2', 'Docker', 
        #    'Registry', 'API', '2']]
        self.test_array3 = [[
            'A', 'Go', 'API', 'client', 'for', 'the', 'v2', 'Docker', 
            'Registry', 'API', '2']]
        # self.test_array4
        # [['A', 'Go', 'API', 'client', 'for', 'the', 'v2', 'Docker', 
        #    'Registry', 'API', 'two']]
        self.test_array4 = [[
            'A', 'Go', 'API', 'client', 'for', 'the', 'v2', 'Docker', 
            'Registry', 'API', 'two']]
        

    def test_tokenize_doc_type_1(self):
        result = self.dp.tokenize_doc(self.doc1)
        self.assertEqual(type(result), numpy.ndarray)
    
    def test_tokenize_doc_type_2(self):
        result = self.dp.tokenize_doc(self.doc1)
        self.assertEqual(type(result[0]), list)
    
    def test_tokenize_doc_contents_1(self):
        result = self.dp.tokenize_doc(self.doc1)
        # self.test_array1
        # ['a', 'easy', 'way', 'to', 'use', 'android', 'sharepreference']
        self.assertEqual(result[0], self.test_array1)

    def test_tokenize_doc_lowercase(self):
        result = self.dp.tokenize_doc(self.doc1)
        actual_word = "webworker" # Lowercase from actual WebWorker
        self.assertEqual(result[11][1], actual_word)

    def test_remove_numbers_type(self):
        # self.test_array2
        # [['A', 'Go', 'API', 'client', 'for', 'the', 'v2', 'Docker', 
        #    'Registry', 'API']]
        result = self.dp.remove_numbers(self.test_array2)
        self.assertEqual(type(self.test_array2), type(result))

    def test_remove_numbers_keep_words_with_numbers(self):
        # self.test_array2
        # [['A', 'Go', 'API', 'client', 'for', 'the', 'v2', 'Docker', 
        #    'Registry', 'API']]
        result = self.dp.remove_numbers(self.test_array2)
        self.assertEqual(self.test_array2, result)

    def test_remove_numbers_1(self):
        # self.test_array3
        # [['A', 'Go', 'API', 'client', 'for', 'the', 'v2', 'Docker', 
        #    'Registry', 'API', '2']]
        expected_result = [[
            'A', 'Go', 'API', 'client', 'for', 'the', 'v2', 'Docker', 
            'Registry', 'API']]
        result = self.dp.remove_numbers(self.test_array3)
        self.assertEqual(result, expected_result)
    
    def test_remove_numbers_2(self):
        # self.test_array4
        # [['A', 'Go', 'API', 'client', 'for', 'the', 'v2', 'Docker', 
        #    'Registry', 'API', 'two']]
        result = self.dp.remove_numbers(self.test_array4)
        self.assertEqual(result, self.test_array4)
    
    def test_remove_nltk_stopwords_type(self):
        test_array = [[
            'a', 'boilerplate', 'for', 'a', 'Koa', 'Redux', 'React', 
            'application', 'with', 'Webpack,', 'Mocha', 'and', 'SASS'
        ]]
        expected_result = [[
            'boilerplate', 'Koa', 'Redux', 'React', 'application', 
            'Webpack,', 'Mocha', 'SASS'
        ]]
        result = self.dp.remove_nltk_stopwords(test_array)
        self.assertEqual(expected_result, result)
        
    def test_stopwords(self):
        stop_words = self.dp.stop_words
        self.assertFalse("p" in stop_words)

if __name__ == '__main__':
    unittest.main()
