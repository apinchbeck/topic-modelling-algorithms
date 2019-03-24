class DocsPreprocessor:
    """
    A class used to process text documents in preparation for applying 
    topic modelling algorithms. 

    ...

    Attributes
    ----------
    text_doc : file
        An open txt file that represents, line-by-line, the documents to
        be modelled by the algorithms.

    Methods
    -------


    """
    
    def __init__(self, doc):
        """
        Parameters
        ----------
        doc : file
            An open txt file that represents, line-by-line, the 
            documents to be modelled by the algorithms.
        """
        self.doc = doc