'''
AUTHOR: AMPARO GODOY PASTORE
ASSIGNMENT 1: NLTK 
COURSE: NATURAL LANGUAGE PROCESSING - CAP6640
INSTRUCTOR: DINGDING WANG
DATE: SEPTEMBER 15TH, 2024 - FALL 2024
'''

# importing libraries
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import numpy as np
import pandas as pd # type: ignore


# read files
file = open("document 1.txt", "r")
doc1 = file.read()
file.close()

file = open("document 2.txt", "r")
doc2 = file.read()
file.close()

file = open("document 3.txt", "r")
doc3 = file.read()
file.close()


# ------- PART 2 -------
# tokenize documents into words
tok_doc1 = word_tokenize(doc1)
tok_doc2 = word_tokenize(doc2)
tok_doc3 = word_tokenize(doc3)

# add symbols, like punctuation marks to stopwords list
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)
for p in punctuation:
    stop_words.add(p)

# remove stop words
doc1_filtered = [w for w in tok_doc1 if not w in stop_words]
doc2_filtered = [w for w in tok_doc2 if not w in stop_words]
doc3_filtered = [w for w in tok_doc3 if not w in stop_words]

# conduct stemming
ps = PorterStemmer()

doc1_stemmed = [ps.stem(w) for w in doc1_filtered]
doc2_stemmed = [ps.stem(w) for w in doc2_filtered]
doc3_stemmed = [ps.stem(w) for w in doc3_filtered]

# create a corpus
doc1_ = ' '.join(c for c in doc1_stemmed)
doc2_ = ' '.join(c for c in doc2_stemmed)
doc3_ = ' '.join(c for c in doc3_stemmed)

corpus = [doc1_, doc2_, doc3_]

# ------- PART 3 -------
# calculate td-idf for each word in each document and generate document-word matrix 
# (each element in the matrix is the tf-idf score for a word in a document)

# Tf-idf matrix
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# extract feature names
tfidf_tokens = vectorizer.get_feature_names_out()

# build a pandas data frame
result = pd.DataFrame(
    data = X.toarray(),
    index = ["doc1_", "doc2_", "doc3_"],
    columns = tfidf_tokens
)

print(f"The following is the tf-idf matrix: {result}")

# ------- PART 4 -------
# calculate pairwise cosine sim for the documents

cosim = cosine_similarity(X)

# I'm converting it to a pandas data frame for easier viewing
cosim_df = pd.DataFrame(cosim, 
                        index=["doc1", "doc2", "doc3"],
                        columns=["doc1", "doc2", "doc3"]
)

print(f"The following is the pairwise cosine similarity for the documents: {cosim_df}")