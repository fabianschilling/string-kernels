#-------------------------------------------------------------------------------------
# Description: Method for computing WK kernel
# Author: J Manttari
# Params: doc1, doc2 - (Cleaned) Input text documents to compute inner product for
# Returns: Inner product <doc1, doc2> as defined by WK kernel
#-------------------------------------------------------------------------------------

#TODO: remove imports
import pandas as pd       
from bs4 import BeautifulSoup             
import re
from nltk.corpus import stopwords # Import the stop word list
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import pickle

def wk(doc1, doc2):
    #print "Creating the bag of words...\n"
    clean_docs = [doc1,doc2]
    #defaultanalyzer "word" removes non-chars in preprocessing and tokenizes words. does not remove "markup tokens"
    #stop_words should be "english" if not using clean_input_docs()
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = "english") 

    train_data_features = vectorizer.fit_transform(clean_docs)
    train_data_features = train_data_features.toarray()

    transformer = TfidfTransformer(smooth_idf=False)
    tfidf = transformer.fit_transform(train_data_features)
    tfidf = tfidf.toarray() 
    #print tfidf
    #print "done"
    #print tfidf.shape
    return np.dot(tfidf[0],tfidf[1]) #returns dot product of given 2 documents