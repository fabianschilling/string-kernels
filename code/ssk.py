import numpy as np

def ssk(doc1, doc2, n=2, lam=0.5, norm=True):
    a = list(doc1)
    b = list(doc2)
    Kp = np.zeros((n + 1, len(a), len(b)), dtype=np.float64)
    Kp[0, :] = 1.0
    for i in range(n):
        for j in range(len(a) - 1):
            Kpp = 0.0
            for k in range(len(b) - 1):
                Kpp = lam * (Kpp + lam * int(a[j] == b[k]) * Kp[i, j, k])
                Kp[i+1, j+1, k+1] = lam * Kp[i+1, j, k+1] + Kpp
    K = 0.0
    for i in range(n):
        for j in range(len(a)):
            for k in range(len(b)):
                K += lam * lam * int(a[j] == b[k]) * Kp[i, j, k]
    
    if norm:
        K1 = ssk(doc1, doc1, n=n, lam=lam, norm=False)
        K2 = ssk(doc2, doc2, n=n, lam=lam, norm=False)
        return K / np.sqrt(K1 * K2)

    return K
