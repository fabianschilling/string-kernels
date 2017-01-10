__author__ = 'Fabian Schilling'
__email__ = 'fabsch@kth.se'

import numpy as np
import functools

def ssk(doc1, doc2, n=2, lam=0.5, norm=True):
  """Computes the string subsequence kernel (SSK)
  Args:
    doc1: First document
    doc2: Second document
    n: Maximum length of the subsequence
    lam: Decay factor
    norm: Normalize to equal document lenghts

  Returns:
    Similarity between the two documents
  """
  result = K(doc1, doc2, n, lam)
  if norm:
    return result / np.sqrt(K(doc1, doc1, n, lam) * K(doc2, doc2, n, lam))
  return result


@functools.lru_cache(maxsize=None)
def K(s, t, n, lam):
  if min(len(s), len(t)) < n:
    return 0.0
  jsum = 0.0
  for j in range(len(t)):
    if t[j] == s[-1]:
      jsum += Kp(s[:-1], t[:j], n-1, lam)
  return K(s[:-1], t, n, lam) + lam ** 2 * jsum


@functools.lru_cache(maxsize=None)
def Kp(s, t, n, lam):
  if n == 0:
    return 1.0
  if min(len(s), len(t)) < n:
    return 0.0
  jsum = 0.0
  for j in range(len(t)):
    if t[j] == s[-1]:
      jsum += Kp(s[:-1], t[:j], n-1, lam) * (lam ** (len(t) - (j + 1) + 2))
  return lam * Kp(s[:-1], t, n, lam) + jsum

