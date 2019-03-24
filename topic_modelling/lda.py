# Import required packages
import numpy as np
import logging
#import pyLDAvis.gensim
import json
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)  # To ignore all warnings that arise here to enhance clarity

# from gensim.models.coherencemodel import CoherenceModel
# from gensim.models.ldamodel import LdaModel
# from gensim.corpora.dictionary import Dictionary
# from numpy import array

# # Import dataset
p_df = pd.read_csv('./docs/description.csv')
# # Create sample of 1000 descriptions
# #p_df = p_df.sample(n = 1000)
# # Convert to array
# docs =array(p_df['description'])
# # Define function for tokenize and lemmatizing
# from nltk.stem.wordnet import WordNetLemmatizer
# from nltk.tokenize import RegexpTokenizer
# from nltk.corpus import stopwords

# def docs_preprocessor(docs):
#     tokenizer = RegexpTokenizer(r'\w+')
#     for idx in range(len(docs)):
#         docs[idx] = docs[idx].lower()  # Convert to lowercase.
#         docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

#     # Remove numbers, but not words that contain numbers.
#     docs = [[token for token in doc if not token.isdigit()] for doc in docs]
    
#     # Remove words that are only one character.
#     docs = [[token for token in doc if len(token) > 3] for doc in docs]

#     # Remove NLTK stop words
#     stop_words = set(stopwords.words('English'))
#     docs = [[token for token in doc if not token in stop_words] for doc in docs]
    
#     # Lemmatize all words in documents.
#     lemmatizer = WordNetLemmatizer()
#     docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]
#     return docs

# # Perform function on our document
# docs = docs_preprocessor(docs)
# #Create Biagram & Trigram Models 
# from gensim.models import Phrases
# # Add bigrams and trigrams to docs,minimum count 10 means only that appear 10 times or more.
# bigram = Phrases(docs, min_count=10)
# trigram = Phrases(bigram[docs])

# for idx in range(len(docs)):
#     for token in bigram[docs[idx]]:
#         if '_' in token:
#             # Token is a bigram, add to document.
#             docs[idx].append(token)
#     for token in trigram[docs[idx]]:
#         if '_' in token:
#             # Token is a bigram, add to document.
#             docs[idx].append(token)
# #Remove rare & common tokens 
# # Create a dictionary representation of the documents.
# dictionary = Dictionary(docs)
# dictionary.filter_extremes(no_below=10, no_above=0.2)
# #Create dictionary and corpus required for Topic Modeling
# corpus = [dictionary.doc2bow(doc) for doc in docs]

# def compute_coherence_values(dictionary, corpus, texts, start, limit, step):
#     """
#     Compute c_v coherence for various number of topics

#     Parameters:
#     ----------
#     dictionary : Gensim dictionary
#     corpus : Gensim corpus
#     texts : List of input texts
#     start : The minimum number of topics
#     limit : Max num of topics
#     step : The step size of the topics

#     Returns:
#     -------
#     model_list : List of LDA topic models
#     coherence_values : Coherence values corresponding to the LDA model with respective number of topics
#     """
#     coherence_values = []
#     model_list = []
#     topic_list = []
#     for num_topics in range(start, limit, step):
#         print("num_topics:\t" + str(num_topics))
#         model=LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
#         model_list.append(model)
#         coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
#         coherence_lda = coherencemodel.get_coherence()
#         coherence_values.append(coherence_lda)
#         topic_list.append(model.show_topics(num_topics=num_topics, num_words=20, log=False, formatted=False))
#     return model_list, coherence_values, topic_list

# import matplotlib.pyplot as plt
# import csv

# def main():
#     model_list, coherence_values, topic_list = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=docs, start=20, limit=101, step=20)
    
#     i = 0
#     coherence_data = []
#     for topic in topic_list:
#         num_topics = str(len(topic))
#         row = []
#         row.append(num_topics)
#         row.append(str(coherence_values[i]))
#         coherence_data.append(row)
#         i += 1
#         topwords_data = []
#         for top in topic:
#             t_row = []
#             t_row.append(top)
#             topwords_data.append(t_row)
#         with open('./results-lda/topic-list-' + str(num_topics) + '.csv', 'w', newline='') as csvFile:
#             writer = csv.writer(csvFile)
#             writer.writerows(topwords_data)
#         csvFile.close()

#     with open('./coherence-data.csv', 'w', newline='') as csvFile:
#         writer = csv.writer(csvFile)
#         writer.writerows(coherence_data)
    
#     start = 20
#     limit= 101
#     step= 20
#     # Create graph
#     x = range(start, limit, step)
#     plt.plot(x, coherence_values)
#     plt.xlabel("Num Topics")
#     plt.ylabel("Coherence score")
#     plt.legend(("coherence_values"), loc='best')
#     plt.show()
    

# if __name__ == "__main__":
#     main()