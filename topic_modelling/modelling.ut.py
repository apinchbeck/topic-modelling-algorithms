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
    def test_run_algorithms(self):
        