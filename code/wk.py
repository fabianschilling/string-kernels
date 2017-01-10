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

def wkGmats(trainDocs, testDocs):
    """ Calculates the Kernels for bag of words
        Returns:
          GmatTrain, GmatTest]: Kernel matrix for training, Kernel matrix for testing
    """
    #figure out number of unique features in training set
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = "english") 

    train_data_features = vectorizer.fit_transform(trainDocs)
    train_data_features = train_data_features.toarray()
    
    n_features_train = len(train_data_features[0])
    
    #figure out number of unique features in test set
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = "english") 

    train_data_features = vectorizer.fit_transform(testDocs)
    train_data_features = train_data_features.toarray()
    
    #n_features_test = len(train_data_features[0])
    n_features_test = 400
    #use smallest feature set (otherwise dot product has size mismatch)
    n_features_min = min(n_features_train,n_features_test)
    print("features train: ", n_features_train)
    print("features test: ", n_features_test)
    print("features used: ", n_features_min)
    
    #defaultanalyzer "word" removes non-chars in preprocessing and tokenizes words. does not remove "markup tokens"
    #stop_words should be "english" if not using clean_input_docs()
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 max_features = n_features_min, \
                                 stop_words = "english") 

    train_data_features = vectorizer.fit_transform(trainDocs)
    train_data_features = train_data_features.toarray()

    transformer = TfidfTransformer(smooth_idf=False)
    tfidf = transformer.fit_transform(train_data_features)
    tfidf = tfidf.toarray() 
    #print tfidf
    #print "done"
    #print tfidf.shape
    nTrainDocs = len(tfidf)
    GmatTrain = np.ones((nTrainDocs,nTrainDocs))

    for i in range( 0, nTrainDocs ):
        if( (i+1)%(nTrainDocs/5) == 0 ):
            print("row %d of %d\n" % ( i+1, nTrainDocs )       )
        for j in range(0,nTrainDocs):
            GmatTrain[i][j] = np.dot(tfidf[i], tfidf[j])
            
    n_features_train = len(tfidf[0])
    
    
    #defaultanalyzer "word" removes non-chars in preprocessing and tokenizes words. does not remove "markup tokens"
    #stop_words should be "english" if not using clean_input_docs()
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 max_features = n_features_min, \
                                 stop_words = "english") 

    train_data_features = vectorizer.fit_transform(testDocs)
    train_data_features = train_data_features.toarray()

    transformer = TfidfTransformer(smooth_idf=False)
    tfidfTest = transformer.fit_transform(train_data_features)
    tfidfTest = tfidfTest.toarray() 
    #print tfidf
    #print "done"
    #print tfidf.shape
    nTestDocs = len(tfidfTest)
    GmatTest = np.ones((nTestDocs,nTrainDocs))

    for i in range( 0, nTestDocs ):
        if( (i+1)%(nTestDocs/5) == 0 ):
            print("row %d of %d\n" % ( i+1, nTestDocs )       )
        for j in range(0,nTrainDocs):
            GmatTrain[i][j] = np.dot(tfidfTest[i], tfidf[j])

    return GmatTrain, GmatTest
