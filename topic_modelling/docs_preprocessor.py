# Package imports

import warnings
# To ignore all warnings that arise here to enhance clarity
warnings.filterwarnings('ignore', category=DeprecationWarning)  

from numpy import array
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords # 179 words as of 2019-03-26

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
    remove_numbers(docs)
        Removes numbers from a tokenized numpy.ndarray, but not words
        that contain numbers. 
    remove_nltk_stopwords(docs)
        Removes NLTK stopwords from a tokenized list. 

    """
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
        
    def tokenize_doc(self, doc):
        """
        Splits a csv file into a collection of tokenized words in order
        to process the words for topic modelling.

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
        Removes numbers from a tokenized numpy.ndarray, but not words
        that contain numbers. 

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

    def remove_nltk_stopwords(self, docs):
        """
        Removes NLTK stopwords from a tokenized list. 

        Parameters
        ----------
        docs : list
            An collection of lists, where each list represents one 
            "document" to be used in the modelling, and each item in the
            list is one word from the document. 

        Returns
        -------
        list
            An array of lists that has been stripped of NLTK stop words.
        """
        docs = [[
            token for token in doc if not token in self.stop_words] 
            for doc in docs]
        return docs