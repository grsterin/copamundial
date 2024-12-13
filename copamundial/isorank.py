import numpy as np
import pandas as pd
from .io_utils import compute_adjacency, compute_pairs

def isorank(A1, A2, E, alpha, maxiter = 20, get_R0 = False, get_R1 = False):
    """
    Compute the isorank using the eigendecomposition
    """

    d1 = np.sum(A1, axis = 1).reshape(-1, 1)
    d2 = np.sum(A2, axis = 1).reshape(-1, 1)
    
    P1 = A1 / d1.T
    P2 = A2 / d2.T
    
    E = E / np.sum(E)
    
    d = d1 @ d2.T 
    d = d / (np.sum(d1) * np.sum(d2))
    
    R = (1-alpha) * d + alpha * E
    
    if maxiter <= 0:
        return R
    
    if get_R0:
        R0 = R.copy()
    
    # Reshape R and E
    R = R.T
    E = E.T
    
    for i in range(maxiter):
        R = (1-alpha) * (P2 @ R @ P1.T) + alpha * E
        if get_R1 and i == 0:
            R1 = R.T.copy()
            
    payload = [R.T]
    if get_R1:
        payload = [R1] + payload
    if get_R0:
        payload = [R0] + payload
    return payload

def compute_greedy_assignment(R1, n_align):
    """
    Compute greedy assignment
    """
    aligned = []
    R = R1.copy()
    n_align = min(n_align, *R.shape)
    itr = 1
    while(len(aligned) < n_align):
        itr   += 1
        maxcols = np.argmax(R, axis = 1) # best y ids
        maxid = np.argmax(np.max(R, axis = 1)) # best x id
        maxcol = maxcols[maxid]
        aligned.append((maxid, maxcol))
        R[:, maxcol] = -1
        R[maxid, :]  = -1
    return aligned


def compute_isorank_and_save(ppiA, ppiB, nameA, nameB, matchfile, alpha, n_align, save_loc, **kwargs):
    A1, protAmap = compute_adjacency(pd.read_csv(ppiA, sep = "\t", header = None))
    A2, protBmap = compute_adjacency(pd.read_csv(ppiB, sep = "\t", header = None))
    rprotAmap = {v:k for k, v in protAmap.items()}
    rprotBmap = {v:k for k, v in protBmap.items()}
    pdmatch = pd.read_csv(matchfile, sep = "\t")
    pdmatch = pdmatch.loc[pdmatch[nameA].apply(lambda x : x in protAmap) & pdmatch[nameB].apply(lambda x : x in protBmap), :]
    print(f"[!] {kwargs['msg']}")
    print(f"[!!] \tSize of the matchfile: {len(pdmatch)}")
    E = compute_pairs(pdmatch, protAmap, 
                     protBmap, nameA, nameB)
    
    R0 = isorank(A1, A2, E, alpha, maxiter = -1)
    align = compute_greedy_assignment(R0, n_align)
    aligndf = pd.DataFrame(align, columns = [nameA, nameB])
    aligndf.iloc[:, 0] = aligndf.iloc[:, 0].apply(lambda x : rprotAmap[x])
    aligndf.iloc[:, 1] = aligndf.iloc[:, 1].apply(lambda x : rprotBmap[x])
    aligndf.to_csv(save_loc, sep = "\t", index = None)
    return