from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.models.nmf import Nmf

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
        print(type(self.docs))

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
        dictionary = Dictionary(self.docs)
        dictionary.filter_extremes(no_below=10, no_above=0.2)
        corpus = [dictionary.doc2bow(doc) for doc in self.docs]

        k_values = []
        coherence_values = []
        topic_list = []
        for num_topics in range(kmin, kmax+1, kstep):
            print("num_topics:\t" + str(num_topics))
            model=LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
            coherencemodel = CoherenceModel(model=model, texts=self.docs, dictionary=dictionary, coherence='c_v')
            coherence_lda = coherencemodel.get_coherence()
            coherence_values.append(coherence_lda)
            topic_list.append(model.show_topics(num_topics=num_topics, num_words=20, log=False, formatted=True))
            k_values.append(num_topics)
        return k_values, coherence_values, topic_list