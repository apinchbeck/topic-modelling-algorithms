# Package imports

import warnings
# To ignore all warnings that arise here to enhance clarity
warnings.filterwarnings('ignore', category=DeprecationWarning)  

from numpy import array
from nltk.tokenize import RegexpTokenizer

class DocsPreprocessor:
    """
    A class used to process text documents in preparation for applying 
    topic modelling algorithms. 

    ...

    Attributes
    ----------
    text_doc : file
        An open csv file that represents, line-by-line, the documents to
        be modelled by the algorithms.

    Methods
    -------
    tokenize_doc(doc)
        Splits a csv file into a collection of tokenized words in order
        to process the words for topic modelling. 

    """
    
    #def __init__(self):
        
        
    def tokenize_doc(self, doc):
        """
        Parameters
        ----------
        doc : file
            An open csv file that represents, line-by-line, the 
            documents to be modelled by the algorithms.

        Returns
        -------
        numpy.ndarray
            An array of lists, where each list represents one "document"
            to be used in the modelling, and each item in the list is 
            one word from the document. 
        """
        docs =array(doc['description'])
        tokenizer = RegexpTokenizer(r'\w+')
        for idx in range(len(docs)):
            docs[idx] = docs[idx].lower()  # Convert to lowercase.
            docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.
        return docs

    def remove_numbers(self, docs):
        """
        Parameters
        ----------
        docs : numpy.ndarray
            An array of lists, where each list represents one "document"
            to be used in the modelling, and each item in the list is 
            one word from the document. 

        Returns
        -------
        list
            An array of lists that has been stripped of numbers, where 
            each list represents one "document" to be used in the 
            modelling, and each item in the list is one word from the 
            document.
        """
        # Remove numbers, but not words that contain numbers.
        docs = [[token for token in doc if not token.isdigit()] for doc in docs]
        return docs