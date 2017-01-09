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
