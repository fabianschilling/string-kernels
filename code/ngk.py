__author__ = 'Fabian Schilling'
__email__ = 'fabsch@kth.se'

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def ngk(doc1, doc2, n=2, mode='char', norm=True):
    """ Compute the n-gram kernel for two documents
  Args:
    doc1: First document
    doc2: Second document
    n: Order of the n-gram
    mode: Either 'char' or 'word' (default: 'char')
    norm: Tfidf normalization (default: true)

  Returns:
    Similarity between the two documents
  """

    # Counts the occurences of unique n-grams
    ngrams = CountVectorizer(analyzer=mode, ngram_range=(n, n)).fit_transform([doc1, doc2])

    # Optionally fit a Tfidf transform
    if norm:
        a, b = TfidfTransformer().fit_transform(ngrams).toarray()
    else:
        a, b = ngrams.toarray()

    return np.dot(a, b)

def ngkGmats(trainDocs,testDocs, n=2, mode='char', norm=True):
    """ Compute the n-gram kernel matrices for train and test documents
      Args:
        trainDocs: First document
        testDocs: Second document
        n: Order of the n-gram
        mode: Either 'char' or 'word' (default: 'char')
        norm: Tfidf normalization (default: true)

      Returns:
        [ngkTrainKmat, ngkTestKmat]: pre-computed kernel matrices for training and testing documents
    """
    # Counts the occurences of unique n-grams in train docs
    vectorizer = CountVectorizer(analyzer=mode, ngram_range=(n, n))
    ngrams = vectorizer.fit_transform(trainDocs)
    
    # Optionally fit a Tfidf transform
    if norm:
        featVecsTrain = TfidfTransformer().fit_transform(ngrams).toarray()
    else:
        featVecsTrain = ngrams.toarray()
        
    
    # Counts the occurences of unique n-grams in train docs
    ngrams = vectorizer.transform(testDocs)
    
    # Optionally fit a Tfidf transform
    if norm:
        featVecsTest = TfidfTransformer().fit_transform(ngrams).toarray()
    else:
        featVecsTest = ngrams.toarray()
        
    nTrainDocs = len(featVecsTrain)
    nTestDocs = len(featVecsTest)
    
    ngkTrainKmat = np.ones((nTrainDocs,nTrainDocs))

    for i in xrange( 0, nTrainDocs ):
        for j in xrange(0,nTrainDocs):
            ngkTrainKmat[i][j] = np.dot(featVecsTrain[i], featVecsTrain[j])
    
    ngkTestKmat = np.ones((nTestDocs,nTrainDocs))

    for i in xrange( 0, nTestDocs ):
        for j in xrange(0,nTrainDocs):
            ngkTestKmat[i][j] = np.dot(featVecsTest[i], featVecsTrain[j])
    
    return ngkTrainKmat, ngkTestKmat
