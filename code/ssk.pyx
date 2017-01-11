from __future__ import division
import numpy as np
cimport numpy as np

def K(s, t, int n, double lam):
  cdef int len_s = len(s)
  cdef int len_t = len(t)
  cdef np.ndarray kd = np.zeros((2, len_s + 1, len_t + 1), dtype=np.float64)
  cdef double kdd = 0.0
  cdef double jsum = 0.0

  for m in range(2):
    for i in range(len_s + 1):
      for j in range(len_t + 1):
        kd[m][i][j] = (m + 1) % 2

  for i in range(1, n):
    for j in range(i - 1, len_s):
      kd[i % 2][j][i - 1] = 0.0
    for j in range(i - 1, len_t):
      kd[i % 2][i - 1][j] = 0.0
    for j in range(1, len_s):
      kdd = 0.0
      for m in range(i, len_t):
        if s[j - 1] != t[m - 1]:
          kdd = lam * kdd
        else:
          kdd = lam * (kdd + (lam * kd[(i + 1) % 2][j - 1][m - 1]))
        kd[i % 2][j][m] = lam * kd[i % 2][j - 1][m] + kdd

  for i in range(n, len_s + 1):
    for j in range(n, len_t + 1):
      if s[i - 1] == t[j - 1]:
        jsum += lam * lam * kd[(n - 1) % 2][i - 1][j - 1]

  return jsum


def ssk(doc1, doc2, int n=2, double lam=0.5):
  return K(doc1, doc2, n, lam) / np.sqrt(K(doc1, doc1, n, lam) * K(doc2, doc2, n, lam))
