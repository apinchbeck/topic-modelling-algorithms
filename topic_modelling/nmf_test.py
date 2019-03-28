from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

class NMF:

    """
    A class used to compute NMF stuff. 

    ...

    Attributes
    ----------
    

    Methods
    -------
    

    """

    def __init__(self, docs_in):
        """
        """
        self.token_docs = docs_in
        self.docs = []
        for d in docs_in:
            row = ""
            for word in d:
                row += word + " "
            self.docs.append(row.strip())

    def compute_coherence_values(self, kmin, kmax, kstep):
        """
        Compute c_v coherence for various number of topics

        Parameters:
        ----------
        kmin : The minimum number of topics
        kmax : Max num of topics
        kstep : The step size of the topics

        Returns:
        -------
        k_values: The number of topics used. 
        coherence_values : Coherence values corresponding to the LDA model with respective number of topics
        topic_list: The list of topics. 
        """
        k_values = []
        coherence_values = []
        topic_list = []

        # feed NMF with tf-idf 
        tfidf_vectorizer = TfidfVectorizer()
        tfidf = tfidf_vectorizer.fit_transform(self.docs)
        tfidf_feature_names = tfidf_vectorizer.get_feature_names()
        
        
        return k_values, coherence_values, topic_list