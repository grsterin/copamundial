import numpy as np

def compute_adjacency(df, nodemap = None):
    if nodemap == None:
        nodeset = set(df[0]).union(set(df[1]))
        nodemap = {k: i for i, k in enumerate(nodeset)}
    n = len(nodemap)
    A = np.zeros((n, n))
    for p, q in df.values:
        p_, q_ = [nodemap[p], nodemap[q]]
        A[p_, q_] = 1
        A[q_, p_] = 1
    return A, nodemap


def compute_pairs(df, nmapA, nmapB, orgA, orgB):
    df = df.loc[:, [orgA, orgB, "score"]]
    
    df[orgA] = df[orgA].apply(lambda x: nmapA[x])
    df[orgB] = df[orgB].apply(lambda x: nmapB[x])
    print(df)
    
    m, n = len(nmapA), len(nmapB)
    A = np.zeros((m, n))
    
    for p, q, v in df.values:
        A[int(p + 0.25), int(q + 0.25)] = v
    return A