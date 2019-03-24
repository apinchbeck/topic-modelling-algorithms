import os.path
import pandas as pd
from numpy import array

# Import dataset
p_df = pd.read_csv('./description.csv')
# Create sample of 1000 descriptions
#p_df = p_df.sample(n = 1000)
# Convert to array
docs =array(p_df['description'])
# Define function for tokenize and lemmatizing
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

def docs_preprocessor(docs):
    tokenizer = RegexpTokenizer(r'\w+')
    for idx in range(len(docs)):
        docs[idx] = docs[idx].lower()  # Convert to lowercase.
        docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

    # Remove numbers, but not words that contain numbers.
    docs = [[token for token in doc if not token.isdigit()] for doc in docs]
    
    # Remove words that are only one character.
    docs = [[token for token in doc if len(token) > 3] for doc in docs]

    # Remove NLTK stop words
    stop_words = set(stopwords.words('English'))
    docs = [[token for token in doc if not token in stop_words] for doc in docs]
    
    """ # Lemmatize all words in documents.
    lemmatizer = WordNetLemmatizer()
    docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs] """
    
    return docs

# Perform function on our document
for d in docs[:10]:
    print(d)
print()
docs = docs_preprocessor(docs)
for d in docs[:10]:
    print(d)