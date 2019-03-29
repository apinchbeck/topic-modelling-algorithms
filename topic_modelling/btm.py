from sklearn.feature_extraction.text import CountVectorizer
from biterm.utility import vec_to_biterms, topic_summuary
from biterm.cbtm import oBTM
import numpy as np

class BTM:
    """
    https://pypi.org/project/biterm/
    """

    def __init__(self, docs_in):
        self.token_docs = docs_in
        self.docs = []
        for d in docs_in:
            row = ""
            for word in d:
                row += word + " "
            self.docs.append(row.strip())

    def compute_values(self, kmin, kmax, kstep):
        # vectorize doc
        vec = CountVectorizer()
        X = vec.fit_transform(self.docs)
        
        # get vocabulary and biterms from docs
        vocab = np.array(vec.get_feature_names())
        biterms = vec_to_biterms(X)

        # create a BTM and pass the biterms to train it
        btm = oBTM(num_topics = 20, V = vocab)
        topics = btm.fit_transform(biterms, iterations=100)
        topic_summuary(btm.phi_wz.T, X, vocab, 10)
        