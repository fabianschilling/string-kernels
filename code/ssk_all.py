import numpy as np
import sys
import math
from modshogun import StringCharFeatures, BinaryLabels
from modshogun import LibSVM, SubsequenceStringKernel, RAWBYTE
from modshogun import ErrorRateMeasure


def get_kernel_shogun_lib(docs, n, lam):
    # print docs
    feats_train=StringCharFeatures(docs, RAWBYTE)
    # feats_test=StringCharFeatures(testdat, RAWBYTE)

    kernel=SubsequenceStringKernel(feats_train, feats_train, 2, 0.5);

    km_train=kernel.get_kernel_matrix()
    # print(km_train)
    # print km_train[0,1]
    return km_train[0, 1]


def get_kernel_shugun (docs, n, lam):
    a = list(docs[0])
    b = list(docs[1])
    Kp = np.zeros((n + 1, len(a), len(b)), dtype=np.float64)
    Kp[0, :] = 1.0
    for i in range(n):
        for j in range(len(a) - 1):
            Kpp = 0.0
            for k in range(len(b) - 1):
                # print "before update: ", i, j, k
                # print Kp
                Kpp = lam * (Kpp + lam * int(a[j] == b[k]) * Kp[i, j, k])
                print a[j], b[k], Kpp, Kp[i+1, j, k+1] 
                # print "lam * (Kpp + lam * int(a[j] == b[k]) * Kp[i, j, k]),  lam * Kp[i+1, j, k+1] + Kpp"
                Kp[i+1, j+1, k+1] = lam * Kp[i+1, j, k+1] + Kpp
    
    print Kp
    K = 0.0
    # for i in range(n):
    for j in range(len(a)):
        for k in range(len(b)):
            K += lam * lam * int(a[j] == b[k]) * Kp[n-1, j, k]
    
    return  K


def get_kernel_nomod2 (docs, k, lam):
    sum = 0
    s = docs[0]
    t = docs[1]
    len_s = len(s)
    len_t = len(t)
    size = k+1, len(s)+1, len(t)+1
    kd = np.zeros(size, dtype=np.float32)

# Dynamic programming 
    for m in xrange(k+1):
        for i in xrange(len_s+1):
            for j in xrange(len_t+1):
                kd[0][i][j] = 1

#  Calculate Kd and Kdd 
#     /* Set the Kd to zero for those lengths of s and t
#     where s (or t) has exactly length i-1 and t (or s)
#     has length >= i-1. L-shaped upside down matrix */
    for i in range(1, k+1):
        for j in range(i -1, len_s):
            kd[i][j][i-1] = 0
        for j in range(i -1, len_t):
            kd[i][i-1][j] = 0
        for j in range(i, len_s):
            kdd = 0
            for m in range(i, len_t):
                if s[j - 1] != t[m - 1]:
                    kdd = lam * kdd
                else: 
                    kdd = lam * (kdd + (lam * kd[(i - 1)][j - 1][m - 1]))     
                # print s[j - 1], t[m - 1], kdd
                kd[i][j][m] = lam * kd[i][j - 1][m] + kdd;

    # for m in range(1, k+1):
    #     for i in range(1, len_s+1):
    #         for j in range(1, len_t+1):
    #             if s[i-1] == t[j-1]:
    #                 sum += lam * lam * kd[(m - 1)][i - 1][j - 1]
# 

    # print kd
    
    for i in range(k, len_s+1):
        for j in range(k, len_t+1):
            if s[i-1] == t[j-1]:
                # print s[i-1], t[j-1], lam * lam * kd[(k - 1)][i - 1][j - 1]
                sum += lam * lam * kd[(k - 1)][i - 1][j - 1]
                # print sum
    return sum

 

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

    print kd

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

def ssk (doc1, doc2, k=2, lam=0.5):
    # size = 2, 2
    # kernel = np.zeros(size, dtype=np.float32)
    norms = np.zeros(2, dtype=np.float32)
    # kernel[:,:] = -1 

    # doc_s = [doc1, doc1]
    # doc_t = [doc2, doc2]
    # # print "c++: "
    # norms[0] = get_kernel_nomod2(doc_s, k, lam)
    # norms[1] = get_kernel_nomod2(doc_t, k, lam)
    # iner_pro = get_kernel_nomod2([doc1, doc2], k, lam)
    # iner_pro = iner_pro/math.sqrt(norms[0] * norms[1])
    # print "c++: ", iner_pro

    # print "norms: "
    # norms[0] = get_kernel_shugun(doc_s, k, lam)
    # norms[1] = get_kernel_shugun(doc_t, k, lam)
    # print "shogun: "
    # iner_pro = get_kernel_shugun([doc1, doc2], k, lam)

    # iner_pro = iner_pro/math.sqrt(norms[0] * norms[1])

    # print "shogun lib: "
    # norms[0] = get_kernel_shugun_lib(doc_s, k, lam)
    # norms[1] = get_kernel_shugun_lib(doc_t, k, lam)
    docs = [str(doc1), str(doc2)]
    iner_pro = get_kernel_shogun_lib(docs, k, lam)

    # iner_pro = iner_pro/math.sqrt(norms[0] * norms[1])

    # print iner_pro
    return iner_pro
    #print kernel

def main(args):
    # docs = ["carcr", "bcbarbbct"]
    # docs = ["cat", "car"]
    ssk(docs[0],docs[1], 7, 0.5)

    
if __name__ == '__main__':
	main(sys.argv)


