import unittest
from docs_preprocessor import DocsPreprocessor

class TestDocsPreprocessor(unittest.TestCase): 
    """
    Test the DocsPreprocessor class.
    """
    
    def setUp(self):
        self.doc1 = open("../docs/description.txt", "r")

    def test_file_type(self):
        d = DocsPreprocessor(self.doc1)
        self.assertIsInstance(d.doc, type(self.doc1))

if __name__ == '__main__':
    unittest.main()
