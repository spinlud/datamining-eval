import io
import argparse
import math
import numpy as np
import scipy as sp
from scipy import io as spio
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
from collections import defaultdict
from collections import OrderedDict


import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


stemmer = PorterStemmer()

doc1 = "Toy Story Pixar A cowboy join an adventure. Adventure, kids children Comedy"
doc2 = "Finding; Nemo ,.Adventure children:,. Pixar"


remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
	


def normalize(text):
	tokens = nltk.word_tokenize(text.lower().translate(remove_punctuation_map))
	stems = [stemmer.stem(t) for t in tokens]
	print stems
	return stems	





doc1 = normalize(doc1)
doc2 = normalize(doc2)

print doc1



# vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')
vectorizer = TfidfVectorizer()



def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]



# print cosine_sim(doc1, doc2)



# doc1 = doc1.lower().translate(None, string.punctuation)
# doc2 = doc2.lower().translate(None, string.punctuation)


# tokens1 = nltk.word_tokenize(doc1)
# tokens2 = nltk.word_tokenize(doc2)

# # tokens1 = [w for w in tokens1 if not w in stopwords.words('english')]
# # tokens2 = [w for w in tokens2 if not w in stopwords.words('english')]

# # print tokens1

# tokens1 = [stemmer.stem(w) for w in tokens1]
# tokens2 = [stemmer.stem(w) for w in tokens2]

# print tokens1


# vect = TfidfVectorizer(min_df=1)
# tfidf = vect.fit_transform([doc1, doc2])

# print (tfidf * tfidf.T).A



