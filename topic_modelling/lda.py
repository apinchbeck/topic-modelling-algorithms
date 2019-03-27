from gensim.corpora.dictionary import Dictionary

class LDA:

    """
    A class used to compute LDA stuff. 

    ...

    Attributes
    ----------
    

    Methods
    -------
    

    """

    def __init__(self, docs):
        """
        """
        self.docs = docs
        self.dictionary = Dictionary(docs)
        self.dictionary.filter_extremes(no_below=10, no_above=0.2)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in docs]

