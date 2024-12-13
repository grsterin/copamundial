import numpy as np

"""
Scoring functions
"""
def _met(golists = {}, go = None, all_go_labels = None, metric = None, **kwargs):
    if len(golists) == 0 or all_go_labels == None or metric == None:
        return None
    
    assert callable(metric)
    
    if go == None or len(go) == 0:
        return 0
    
    godiff = list(all_go_labels.difference(go)) # add the rest of go labels
    goall  = go + godiff
    gopred = np.concatenate([np.linspace(1, 0.5, len(go)), np.zeros(len(godiff),)])
    gotrue = [1 if go in golists else 0 for go in goall]
    sc = metric(gotrue, gopred, **kwargs)
    
    return sc
    
def _mets(prots, pred_go_map, true_go_map, all_go_labels, metric, **kwargs):
    scores = [_met(true_go_map[p], pred_go_map[p], all_go_labels, metric, **kwargs) 
             for p in prots]
    scores = list(filter(lambda x: x is not None, scores))
    return np.average(scores)


def scoring_fcn(all_go_labels, metric, **kwargs):
    assert metric is not None and all_go_labels is not None
    def scfunc(prots, pred_go_map, true_go_map):
        return _mets(prots, pred_go_map, true_go_map, all_go_labels, metric, **kwargs)
    return scfunc
    

def topk_acc(golists = {}, go = None, k = 1):
    if len(golists) == 0:
        return None
    elif go == None:
        return 0
    else:
        go_k = set(go[: min(k, len(go))])
        return len(go_k.intersection(golists)) / k

    
def topk_accs(prots, pred_go_map, true_go_map, k = 1):
    accs = [topk_acc(true_go_map[p], pred_go_map[p], k) for p in prots]
    accs = list(filter(lambda x : x is not None, accs))
    return np.average(accs)


def compute_metric(prediction_func, scoring_func, allprots, true_go_map, kfold = 5, seed = 42):
    np.random.seed(seed)
    permprots = np.random.permutation(allprots)
    blocksize = int(len(allprots) / kfold)
    scores = []
    for i in range(kfold):
        predictprots = permprots[i * blocksize: (i+1) * blocksize]
        trainprots = np.concatenate([permprots[: i * blocksize], permprots[(i+1) * blocksize : ]])
        go_map_training = {tprots: true_go_map[tprots] for tprots in trainprots}
        go_map_training.update({pprots : -1 for pprots in predictprots})
        pred_map = prediction_func(go_map_training)
        scores.append(scoring_func(predictprots, pred_map, true_go_map))
    return scores, np.average(scores)


def predict_dsd(D_mat, train_go_maps, k = 10):
    predprot = [x for x in train_go_maps if train_go_maps[x] == -1]
    D_mat1 = D_mat.copy()
    D_mat1[range(len(D_mat)), range(len(D_mat))] = np.inf
    D_mat1[:, predprot] = np.inf
    sortedD = np.argsort(D_mat1, axis = 1)[:, 1:k+1]
    def vote(neighbors, go_maps):
        gos = {}
        for n in neighbors:
            for g in go_maps[n]:
                if g not in gos:
                    gos[g] = 0
                gos[g] += 1 
        return sorted(gos, key = lambda x : gos[x], reverse=True)
    for p in predprot:
        train_go_maps[p] = vote(sortedD[p], train_go_maps)
    return train_go_maps


def predict_dsd_mundo(D_mat, D_other_species, train_go_maps, go_other, k = 10, k_other = 20, weight_other = 0.4):
    predprot = [x for x in train_go_maps if train_go_maps[x] == -1]
    D_mat1 = D_mat.copy()
    D_other = D_other_species.copy()
    D_mat1[range(len(D_mat)), range(len(D_mat))] = np.inf
    D_mat1[:, predprot] = np.inf
    sortedD = np.argsort(D_mat1, axis = 1)[:, 1: k+1]
    sortedDoth = np.argsort(D_other, axis = 1)[:, 1: k_other+1]
    def vote(neighbors, oth_neighbors,  go_maps):
        gos = {}
        for n in neighbors:
            for g in go_maps[n]:
                if g not in gos:
                    gos[g] = 0
                gos[g] += 1 
        for n in oth_neighbors:
            for g in go_other[n]:
                if g not in gos:
                    gos[g] = 0
                gos[g] += weight_other  
        return sorted(gos, key = lambda x : gos[x], reverse=True)
    for p in predprot:
        train_go_maps[p] = vote(sortedD[p], sortedDoth[p], train_go_maps)
    return train_go_maps


def dsd_func(D_mat, k = 10):
    def pred(train_go_maps):
        return predict_dsd(D_mat, train_go_maps, k = k)
    return pred


def dsd_func_mundo(D_mat, D_other, go_other, k = 10, k_other = 20, weight_other = 0.4):
    def pred(train_go_maps):
        return predict_dsd_mundo(D_mat, D_other, train_go_maps, go_other, k, k_other, weight_other)
    return pred
