from DocsPreprocessor import DocsPreprocessor

import unittest

class TestDocsPreprocessor(unittest.TestCase): 

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')
        print("okay")

if __name__ == '__main__':
    unittest.main()
