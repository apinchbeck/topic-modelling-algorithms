# Package imports

import warnings
# To ignore all warnings that arise here to enhance clarity
warnings.filterwarnings('ignore', category=DeprecationWarning)  

from numpy import array

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords # 179 words as of 2019-03-26
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet

from gensim.models import Phrases
from gensim.models.phrases import Phraser

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
    lemmatize_doc(docs)
        Lemmatizes a list of lists. 
    get_wordnet_pos(word)
        Map POS tag to first character lemmatize() accepts.
    append_bigrams_and_trigrams(docs)
        Turns common phrases into bigrams and trigrams.

    """
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def process(self, docs):
        """
        Fully processes a csv doc:
            - tokenizes
            - removes numbers
            - removes nltk stop words
            - lemmatizes
            - creates bigrams and trigrams
        """
        result = self.tokenize_doc(docs)
        result = self.remove_numbers(result)
        result = self.remove_nltk_stopwords(result)
        result = self.lemmatize_doc(result)
        result = self.append_bigrams_and_trigrams(result)
        return result
    
    def append_bigrams_and_trigrams(self, docs):
        """
        Turns common phrases into bigrams and trigrams.

        Methodology taken from:
        https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
        """
        
        bigram = Phrases(docs, min_count=5, threshold=100)
        bigram_mod = Phraser(bigram)
        trigram = Phrases(bigram[docs], threshold=100)
        trigram_mod = Phraser(trigram)
        
        bigrams = [bigram_mod[doc] for doc in docs]
        trigrams = [trigram_mod[bigram_mod[doc]] for doc in docs]
        
        return trigrams

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

    def lemmatize_doc(self, docs):
        """
        Lemmatizes a list of lists. 

        Parameters
        ----------
        docs : list
            An collection of lists, where each list represents one 
            "document" to be used in the modelling, and each item in the
            list is one word from the document. 

        Returns
        -------
        list
            An array of lists that has been lemmatized.
        """
        docs = [[
            self.lemmatizer.lemmatize(w, self.get_wordnet_pos(w)) for w in doc]
             for doc in docs]
        return docs

    def get_wordnet_pos(self, word):
        """
        Map POS tag to first character lemmatize() accepts.
        
        Method taken from:
        https://www.machinelearningplus.com/nlp/lemmatization-examples-python/

        """
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)
        
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

    