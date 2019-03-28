import matplotlib
import matplotlib.pyplot as plt
plt.style.use("ggplot")
matplotlib.rcParams.update({"font.size": 14})

from nmf import NMF
from lda import LDA
from docs_preprocessor import DocsPreprocessor

class Modelling:
    """
    A class used to do STTM.

    ...

    Attributes
    ----------

    Methods
    -------

    """
    def __init__(self, docs):
        self.dp = DocsPreprocessor()
        self.docs = self.dp.process(docs)
        self.nmf = NMF(self.docs)
        self.lda = LDA(self.docs)
    
    def run_algorithms(self, kmin, kmax, kstep, top):
        self.run_lda(kmin, kmax, kstep)

    def run_lda(self, kmin, kmax, kstep):
        k_values, coherence_values, topic_list =                               \
            self.lda.compute_coherence_values(kmin, kmax, kstep)
        # The next returns the best # of topics and saves the plot
        best_k = self.plot_coherence(k_values, coherence_values, "lda")

    def run_nmf(self, kmin, kmax, kstep, top):
        k_values, coherence_values = self.nmf.process_models(
            kmin, kmax, kstep, top)
        best_k = self.plot_coherence(k_values, coherence_values, "nmf")

    def plot_coherence(self, k_values, coherences, name):
        fig = plt.figure(figsize=(13,7))
        # create the line plot
        ax = plt.plot( k_values, coherences )
        plt.xticks(k_values)
        plt.xlabel("Number of Topics")
        plt.ylabel("Mean Coherence")
        # add the points
        plt.scatter( k_values, coherences, s=120)
        # find and annotate the maximum point on the plot
        ymax = max(coherences)
        xpos = coherences.index(ymax)
        best_k = k_values[xpos]
        plt.annotate( 
            "k=%d" % best_k, xy=(best_k, ymax), xytext=(
                best_k, ymax), textcoords="offset points", fontsize=16)
        # save the plot
        plt.savefig(name + "-coherence.jpg")
        return best_k
        
