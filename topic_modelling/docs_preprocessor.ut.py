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
        self.description = pd.read_csv("docs/description.csv")
        self.description_1000 = pd.read_csv("docs/description_1000.csv")
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
        
        # self.test_array5
        # [['a', 'boilerplate', 'for', 'a', 'Koa', 'Redux', 'React', 
        # 'application', 'with', 'Webpack,', 'Mocha', 'and', 'SASS']]
        self.test_array5 = [[
            'a', 'boilerplate', 'for', 'a', 'Koa', 'Redux', 'React', 
            'application', 'with', 'Webpack,', 'Mocha', 'and', 'SASS'
        ]]

        # self.test_array6
        # 'Many', 'corpora', 'have', 'better', 'values', 'than', 
        # 'you', 'think'
        self.test_array6 = [[
            'Many', 'corpora', 'have', 'better', 'values', 'than', 
            'you', 'think'
        ]]
        
    def test_process(self):
        result = self.dp.process(self.description_1000)
        count_front_end = 0
        for row in result:
            for r in row:
                if r == "front_end": count_front_end += 1
        # There are 6 instances of either "front end" or "front-end"
        self.assertTrue(count_front_end == 6)
    
    def test_append_bigrams_and_trigrams(self):
        result = self.dp.tokenize_doc(self.description_1000)
        result = self.dp.remove_numbers(result)
        result = self.dp.remove_nltk_stopwords(result)
        result = self.dp.lemmatize_doc(result)
        result = self.dp.append_bigrams_and_trigrams(result)
        count_open_source = 0
        for row in result:
            for r in row:
                if r == "open_source": count_open_source += 1
        # There are 17 instances of either "open source" or "open-source"
        self.assertTrue(count_open_source == 17)

    def test_tokenize_doc_type_1(self):
        result = self.dp.tokenize_doc(self.description)
        self.assertEqual(type(result), numpy.ndarray)
    
    def test_tokenize_doc_type_2(self):
        result = self.dp.tokenize_doc(self.description)
        self.assertEqual(type(result[0]), list)
    
    def test_tokenize_doc_contents_1(self):
        result = self.dp.tokenize_doc(self.description)
        # self.test_array1
        # ['a', 'easy', 'way', 'to', 'use', 'android', 'sharepreference']
        self.assertEqual(result[0], self.test_array1)

    def test_tokenize_doc_lowercase(self):
        result = self.dp.tokenize_doc(self.description)
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
    
    def test_remove_nltk_stopwords(self):
        # self.test_array5
        # [['a', 'boilerplate', 'for', 'a', 'Koa', 'Redux', 'React', 
        # 'application', 'with', 'Webpack,', 'Mocha', 'and', 'SASS']]
        expected_result = [[
            'boilerplate', 'Koa', 'Redux', 'React', 'application', 
            'Webpack,', 'Mocha', 'SASS'
        ]]
        result = self.dp.remove_nltk_stopwords(self.test_array5)
        self.assertEqual(expected_result, result)
    
    def test_remove_nltk_stopwords_type(self):
        # self.test_array5
        # [['a', 'boilerplate', 'for', 'a', 'Koa', 'Redux', 'React', 
        # 'application', 'with', 'Webpack,', 'Mocha', 'and', 'SASS']]
        expected_result = [[
            'boilerplate', 'Koa', 'Redux', 'React', 'application', 
            'Webpack,', 'Mocha', 'SASS'
        ]]
        result = self.dp.remove_nltk_stopwords(self.test_array5)
        self.assertEqual(type(expected_result), type(result))

    def test_lemmatize_doc_type(self):
        # self.test_array6
        # 'React', 'component', 'for', 'inputting', 'numeric', 
        # 'values', 'within', 'a', 'range'
        result = self.dp.lemmatize_doc(self.test_array6)
        self.assertEqual(type(self.test_array6), type(result))

    def test_lemmatize_doc(self):
        # self.test_array6
        # 'Many', 'corpora', 'have', 'better', 'values', 'than', 
        # 'you', 'think'
        expected_result = [[
            'Many', u'corpus', 'have', u'well', u'value', 'than', 
            'you', 'think'
        ]]
        result = self.dp.lemmatize_doc(self.test_array6)
        self.assertEqual(expected_result, result)

if __name__ == '__main__':
    unittest.main()
