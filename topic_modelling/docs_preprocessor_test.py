import unittest
from docs_preprocessor import DocsPreprocessor

class TestDocsPreprocessor(unittest.TestCase): 

    def test_upper(self):
        d = DocsPreprocessor("string")
        self.assertEquals(d.name, "home")

if __name__ == '__main__':
    unittest.main()
