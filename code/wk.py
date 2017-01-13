#-------------------------------------------------------------------------------------
# Description: Method for computing WK kernel
# Author: J Manttari
# Params: doc1, doc2 - (Cleaned) Input text documents to compute inner product for
# Returns: Inner product <doc1, doc2> as defined by WK kernel
#-------------------------------------------------------------------------------------

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
    return np.dot(tfidf[0],tfidf[1]) #returns dot product of given 2 documents

def wkFeatVecs(trainDocs, testDocs):

    n_features_min = 2000
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
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
            
    n_features_train = len(tfidf[0])
    
    #defaultanalyzer "word" removes non-chars in preprocessing and tokenizes words. does not remove "markup tokens"
    #stop_words should be "english" if not using clean_input_docs()
    
    train_data_features = vectorizer.transform(testDocs)
    train_data_features = train_data_features.toarray()

    transformer = TfidfTransformer(smooth_idf=False)
    tfidfTest = transformer.fit_transform(train_data_features)
    tfidfTest = tfidfTest.toarray() 
    #print tfidf
    #print "done"
    #print tfidf.shape
    nTestDocs = len(tfidfTest)
    GmatTest = np.ones((nTestDocs,nTrainDocs))

    # print "Trainmean: ", GmatTrain.mean()
    # print "Testmean: ", GmatTest.mean()
    
    return tfidf, tfidfTest

def wkGmats(trainDocs, testDocs):
    """ Calculates the Kernels for bag of words
        Returns:
          GmatTrain, GmatTest]: Kernel matrix for training, Kernel matrix for testing
    """
    
    #defaultanalyzer "word" removes non-chars in preprocessing and tokenizes words. does not remove "markup tokens"
    #stop_words should be "english" if not using clean_input_docs()
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = "english") 

    train_data_features = vectorizer.fit_transform(trainDocs)
    train_data_features = train_data_features.toarray()

    transformer = TfidfTransformer(smooth_idf=True)
    tfidf = transformer.fit_transform(train_data_features)
    tfidf = tfidf.toarray() 
    #print tfidf
    #print "done"
    #print tfidf.shape
    nTrainDocs = len(tfidf)
    GmatTrain = np.ones((nTrainDocs,nTrainDocs))

    for i in xrange( 0, nTrainDocs ):
        for j in xrange(0,nTrainDocs):
            GmatTrain[i][j] = np.dot(tfidf[i], tfidf[j])
            
    n_features_train = len(tfidf[0])
    
    train_data_features = vectorizer.transform(testDocs)
    train_data_features = train_data_features.toarray()

    transformer = TfidfTransformer(smooth_idf=True)
    tfidfTest = transformer.fit_transform(train_data_features)
    tfidfTest = tfidfTest.toarray() 
    #print tfidf
    #print "done"
    #print tfidf.shape
    nTestDocs = len(tfidfTest)
    GmatTest = np.ones((nTestDocs,nTrainDocs))

    for i in xrange( 0, nTestDocs ):
        for j in xrange(0,nTrainDocs):
            GmatTest[i][j] = np.dot(tfidfTest[i], tfidf[j])

    # print "Trainmean: ", GmatTrain.mean()
    # print "Testmean: ", GmatTest.mean()
    
    return GmatTrain, GmatTest
