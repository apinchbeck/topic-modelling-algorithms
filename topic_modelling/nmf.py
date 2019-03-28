import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)  # To ignore all warnings that arise here to enhance clarity
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import decomposition
import operator
import gensim
from itertools import combinations

class NMF:

    """
    A class used to computer NMF stuff.

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
    
    def process_models(self, kmax, kmin, kstep, top):
        k_values = []
        coherences = []
        topic_models = self.run_topic_models(kmax, kmin, kstep)
        A, terms = self.vectorize()
        for (k, W, H) in topic_models:
            # Get all of the topic descriptors - the term_rankings, based on top 10 terms
            term_rankings = []
            for topic_index in range(k):
                term_rankings.append( self.get_descriptor( terms, H, topic_index, top ) )
            # Now calculate the coherence based on our Word2vec model
            k_values.append( k )
            coherences.append( self.calculate_coherence(self.create_word_embedding_model(), term_rankings ) )
            # print("K=%02d: Coherence=%.4f" % ( k, coherences[-1] ) )
        return k_values, coherences
    
    def calculate_coherence( self, w2v_model, term_rankings ):
        overall_coherence = 0.0
        for topic_index in range(len(term_rankings)):
            # check each pair of terms
            pair_scores = []
            for pair in combinations( term_rankings[topic_index], 2 ):
                pair_scores.append( w2v_model.similarity(pair[0], pair[1]) )
            # get the mean for all pairs in this topic
            topic_score = sum(pair_scores) / len(pair_scores)
            overall_coherence += topic_score
        # get the mean score across all topics
        return overall_coherence / len(term_rankings)

    def create_word_embedding_model(self):
        w2v_model = gensim.models.Word2Vec(self.token_docs, size=500, min_count = 5, sg=1)
        #print(type(w2v_model))
        #print( "Model has %d terms" % len(w2v_model.wv.vocab) )
        return w2v_model
    
    def run_topic_models(self, kmin, kmax, kstep):
        topic_models = []
        for k in range(kmin, kmax+1, kstep):
            W, H = self.create_model(k)
            topic_models.append((k, W, H))
        return topic_models

    
    def create_model(self, k):
        A, terms = self.vectorize()
        model = decomposition.NMF( init="nndsvd", n_components=k ) 
        # apply the model and extract the two factor matrices
        W = model.fit_transform( A )
        H = model.components_
        return W, H
    
    def get_descriptor( self, terms, H, topic_index, top ):
        # reverse sort the values to sort the indices
        top_indices = np.argsort( H[topic_index,:] )[::-1]
        # now get the terms corresponding to the top-ranked indices
        top_terms = []
        for term_index in top_indices[0:top]:
            top_terms.append( terms[term_index] )
        return top_terms
    
    def vectorize(self):
        vectorizer = TfidfVectorizer(min_df = 5)
        A = vectorizer.fit_transform(self.docs)
        terms = vectorizer.get_feature_names()
        ranking = self.rank_terms( A, terms )
        #for i, pair in enumerate( ranking[0:20] ):
        #    print( "%02d. %s (%.2f)" % ( i+1, pair[0], pair[1] ) )
        return A, terms
    
    def rank_terms( self, A, terms ):
        # get the sums over each column
        sums = A.sum(axis=0)
        # map weights to the terms
        weights = {}
        for col, term in enumerate(terms):
            weights[term] = sums[0,col]
        # rank the terms by their weight over all documents
        return sorted(weights.items(), key=operator.itemgetter(1), reverse=True)