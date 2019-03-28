import matplotlib
import matplotlib.pyplot as plt
plt.style.use("ggplot")
matplotlib.rcParams.update({"font.size": 14})

from nmf import NMF
from lda import LDA
from docs_preprocessor import DocsPreprocessor

import csv
from os.path import join as pjoin
# Set the save path for your own machine
save_path = "C:\\Users\\angiepin\\Topic Modelling\\results\\"

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

    def run_lda(self, kmin, kmax, kstep, n):
        k_values, coherence_values, topic_list =                               \
            self.lda.compute_coherence_values(kmin, kmax, kstep)
        # The next returns the best # of topics and saves the plot
        file_path = save_path + "lda\\"
        
        for model_run in range(1, n + 1):
            best_k = self.plot_coherence(k_values, coherence_values, file_path + str(model_run) + "-")
            self.write_lda_data(topic_list, coherence_values, file_path + str(model_run) + "-")

    def run_nmf(self, kmin, kmax, kstep, top):
        k_values, coherence_values = self.nmf.process_models(
            kmin, kmax, kstep, top)
        best_k = self.plot_coherence(k_values, coherence_values, "nmf")
        

    def write_nmf_data(self, coherence_values):
        """
        """

    def write_lda_data(self, topic_list, coherence_values, file_path):
        """
        """
        i = 0
        coherence_data = []
        for topic in topic_list:
            num_topics = str(len(topic))
            row = []
            row.append(num_topics)
            row.append(str(coherence_values[i]))
            coherence_data.append(row)
            i += 1
            topwords_data = []
            for top in topic:
                t_row = []
                t_row.append(top)
                topwords_data.append(t_row)
            with open(
                file_path + 'lda-topic-list-' + str(
                    num_topics) + '.csv', 'w', newline='') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerows(topwords_data)
            csvFile.close()

        with open(file_path + 'coherence-data.csv', 'w', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(coherence_data)

    def plot_coherence(self, k_values, coherences, file_path):
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
        plt.savefig(file_path + "lda-coherence.png")
        return best_k
