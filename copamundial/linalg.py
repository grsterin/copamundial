import numpy as np
from scipy import linalg
from scipy.linalg import eig


def compute_k_svd(M, k):
    U, s, _ = linalg.svd(M)
    s = np.sqrt(s[:k])
    U = U[:, :k]
    return np.multiply(U, s)


def compute_k_svd_uv(M):
    U, s, V = linalg.svd(M)
    s = np.sqrt(s)
    U1 = np.multiply(U, s)
    V1 = np.multiply(V, s) # M = U1 V1^*
    Urowsum = np.sum(U1, axis = 0)
    Urowsort = np.argsort(-Urowsum)
    U1 = U1[Urowsort, :]
    V1 = V1[Urowsort, :]
    print(Urowsum[:100])
    return U1, V1
        

def compute_k_eig(M, k = None):
    eigenvalues, eigenvectors = eig(M)
    if k == None: 
        k = eigenvectors.shape[0]
    ids = np.argsort(eigenvalues)[:k]
    return eigenvectors[ids] * np.sqrt(eigenvalues[ids])