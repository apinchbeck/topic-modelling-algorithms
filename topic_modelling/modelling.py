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

    def run_lda(self, kmin, kmax, kstep):
        k_values, coherence_values, topic_list = self.lda.compute_coherence_values(kmin, kmax, kstep)
        print(k_values)
        print()
        print(coherence_values)

    def run_nmf(self, kmin, kmax, kstep, top):
        k_values, coherences = self.nmf.process_models(kmin, kmax, kstep, top)

    def plot_algorithm(self, k_values, coherences):
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
        plt.annotate( "k=%d" % best_k, xy=(best_k, ymax), xytext=(best_k, ymax), textcoords="offset points", fontsize=16)
        # show the plot
        plt.show()
