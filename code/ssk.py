import numpy as np
import sys
import math

def get_kernel (docs, k, lam):
    sum = 0
    s = docs[0]
    t = docs[1]
    len_s = len(s)
    len_t = len(t)
    size = 2, len(s)+1, len(t)+1
    kd = np.zeros(size, dtype=np.float32)

# Dynamic programming 
    for m in xrange(2):
        for i in xrange(len_s+1):
            for j in xrange(len_t+1):
                kd[m][i][j] = (m + 1) % 2

#  Calculate Kd and Kdd 
#     /* Set the Kd to zero for those lengths of s and t
#     where s (or t) has exactly length i-1 and t (or s)
#     has length >= i-1. L-shaped upside down matrix */
    for i in range(1, k):
        for j in range(i -1, len_s):
            kd[i % 2][j][i-1] = 0
        for j in range(i -1, len_t):
            kd[i % 2][i-1][j] = 0
        for j in range(i, len_s):
            kdd = 0
            for m in range(i, len_t):
                # print s[j - 1], t[m - 1]
                if s[j - 1] != t[m - 1]:
                    kdd = lam * kdd
                else:
                    kdd = lam * (kdd + (lam * kd[(i + 1) % 2][j - 1][m - 1]))     
                kd[i % 2][j][m] = lam * kd[i % 2][j - 1][m] + kdd;


    for i in range(k, len_s+1):
        for j in range(k, len_t+1):
            if s[i-1] == t[j-1]:
                sum += lam * lam * kd[(k - 1)%2][i - 1][j - 1]
    return sum



def compute_kernel_matrix (docs, lam, k, kernel, norms):
    for i in xrange(2):
        for j in xrange(2):
            if kernel[i][j] == -1:
                kernel[i][j] = get_kernel(docs, k, lam)
                kernel[i][j] = kernel[i][j]/math.sqrt(norms[0] * norms[1])
                kernel[j][i] = kernel[i][j]

    return kernel

def ssk (doc1,doc2, k=2, lam=0.5):
    # if len(s)==0 or len(t) == 0:
        # return wrong
    # size = 2, len(s), len(t)
    print lam, k

    size = 2, 2
    kernel = np.zeros(size, dtype=np.float32)
    norms = np.zeros(2, dtype=np.float32)
    kernel[:,:] = -1 

    doc_s = [doc1, doc1]
    doc_t = [doc2, doc2]
    norms[0] = get_kernel(doc_s, k, lam)
    norms[1] = get_kernel(doc_t, k, lam)

    for i in xrange(2):
        kernel[i,i] = 1

    kernel = compute_kernel_matrix([doc1, doc2], lam, k, kernel, norms);
    #print kernel

def main(args):
    docs = ["science is organized knowledge", "wisdom is organized life"]
    # s, t = "cat", "card"
    ssk(docs[0],docs[1], 0.5, 2)

    
if __name__ == '__main__':
	main(sys.argv)


