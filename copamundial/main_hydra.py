from __future__ import annotations
import argparse
import glidetools.algorithm.dsd as dsd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import svd
import os
from copamundial.data import CuratedData, PredictData
from copamundial.model import SpeciesTranslationModel
from copamundial.isorank import isorank, compute_greedy_assignment
from copamundial.predict_score import topk_accs, compute_metric, dsd_func, dsd_func_mundo, scoring_fcn
import re
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import networkx as nx
import pickle as pkl
import sys
from omegaconf import OmegaConf, DictConfig
import hydra
import logging


def configure_logger(logfile):
    logging.basicConfig(
        level=logging.DEBUG, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler()
        ]
    )
    return


def compute_pairs(df, nmapA, nmapB, speciesA, speciesB):
    """
    Required for ISORANK computation
    """
    df = df.loc[:, [speciesA, speciesB, "score"]]

    df[speciesA] = df[speciesA].apply(lambda x: nmapA[x])
    df[speciesB] = df[speciesB].apply(lambda x: nmapB[x])

    m, n = len(nmapA), len(nmapB)
    E = np.zeros((m, n))

    for p, q, v in df.values:
        E[int(p + 0.25), int(q + 0.25)] = v
    return E


def get_scoring(metric, all_go_labels=None, **kwargs):
    """
    Evaluation methods
    """
    acc = re.compile(r'top-([0-9]+)-acc')
    match_acc = acc.match(metric)
    if match_acc:
        k = int(match_acc.group(1))
        def score(prots, pred_go_map, true_go_map):
            return topk_accs(prots, pred_go_map, true_go_map, k=k)
        return score
    else:
        if metric == "aupr":
            met = average_precision_score
        elif metric == "auc":
            met = roc_auc_score
        elif metric == "f1max":
            def f1max(true, pred):
                pre, rec, _ = precision_recall_curve(true, pred)
                f1 = (2 * pre * rec) / (pre + rec + 1e-7)
                return np.max(f1)
            met = f1max
        sfunc = scoring_fcn(all_go_labels, met, **kwargs)
    return sfunc


def compute_isorank_and_save(Aa, Ab, mapA, mapB, matchfile, speciesA, speciesB, 
                            no_landmarks, isorank_alpha, isorankfile):
    """
    Function that takes the PPI adjacency matrices and sequence similarity matrices 
    from two species and uses it to compute the one-to-one network alignment

    Aa, Ab -> the numpy.ndarray adjacency matrices for two species
    mapA, mapB -> the protein name to index dictionaries for the two species adjacency 
                  matrices
    matchfile -> the location of a tsv file that lists the pairwise sequence similarity
                 scores of proteins. The tsv file should be of the form:
                    <speciesA>  <speciesB> <score>
                    protA1       protB1     0.65
                            ................
    speciesA, speciesB -> the name of the species, specified in the matchfile header
    no_landmarks -> the number of landmarks to select
    isorank_alpha -> An ISORANK parameter
    isorank_file -> output ISORANK one-to-one alignment file
    """
    rmapA = {v: k for k, v in mapA.items()}
    rmapB = {v: k for k, v in mapB.items()}

    pdmatch = pd.read_csv(matchfile, sep="\t")
    pdmatch = pdmatch.loc[
              pdmatch[speciesA].apply(lambda x: x in mapA) & pdmatch[speciesB].apply(lambda x: x in mapB),
              :]

    logging.info(f"[!!] \tSize of the matchfile: {len(pdmatch)}")

    E = compute_pairs(pdmatch, mapA,
                      mapB, speciesA, speciesB)

    R0 = isorank(Aa, Ab, E, isorank_alpha, maxiter=-1)
    align = compute_greedy_assignment(R0, no_landmarks)
    aligndf = pd.DataFrame(align, columns=[speciesA, speciesB])
    aligndf.iloc[:, 0] = aligndf.iloc[:, 0].apply(lambda x: rmapA[x])
    aligndf.iloc[:, 1] = aligndf.iloc[:, 1].apply(lambda x: rmapB[x])
    aligndf.to_csv(isorankfile, sep="\t", index=None)
    return


def compute_dsd_dist(ppifile, dsdfile, dsd_threshold):
    """
    computes the DSD matrices
    """
    if dsdfile is not None and os.path.exists(dsdfile):
        logging.info(f"DSD file already computed <- {dsdfile}")
        with open(dsdfile, "rb") as cfdsd:
            package = pkl.load(cfdsd)
            DSDdist = package["dsd_dist"]
            protmap = package["json"]
            Asub = package["A"]
        if dsd_threshold > 0:
            DSDdist = np.where(DSDdist > dsd_threshold,
                               dsd_threshold,
                               DSDdist)
        return DSDdist, Asub, protmap
    else:
        logging.info(f"Computing DSD file -> {dsdfile}")
        ppdf = pd.read_csv(ppifile, sep="\t", header=None)
        Gpp = nx.from_pandas_edgelist(ppdf, source=0,
                                      target=1)
        ccs = max(nx.connected_components(Gpp), key=len)
        Gpps = Gpp.subgraph(ccs)
        Asub = nx.to_numpy_array(Gpps)
        protmap = {k: i for i, k in enumerate(list(Gpps.nodes))}
        DSDemb = dsd.compute_dsd_embedding(Asub,
                                           is_normalized=False)
        DSDdist = squareform(pdist(DSDemb))
        with open(dsdfile, "wb") as cfdsd:
            pkl.dump({
                "A": Asub,
                "dsd_dist": DSDdist,
                "dsd_emb": DSDemb,
                "json": protmap
            }, cfdsd)
        if dsd_threshold > 0:
            DSDdist = np.where(DSDdist > dsd_threshold,
                               dsd_threshold, DSDdist)
        return DSDdist, Asub, protmap


def compute_svd(svdfile, dsddist, protmap, svd_r):
    """
    svd_r => how many SVD components to select?
    """
    if os.path.exists(svdfile):
        logging.info(f"SVD files already computed <- {svdfile}")
        with open(svdfile, "rb") as svdf:
            package = pkl.load(svdf)
            U = package["U"]
            V = package["V"]
            s = package["s"]
        U = U[:, :svd_r]
        V = V[:svd_r, :]
        s = s[:svd_r]
        ssqrt = np.sqrt(s)
        Ud = U * ssqrt[None, :]  # [n, svd]
        Vd = V * ssqrt[:, None]  # [svd, n]
        return Ud, Vd
    else:
        logging.info(f"Computing the SVD file -> {svdfile}")
        U, s, V = svd(dsddist)
        with open(svdfile, "wb") as svdf:
            pkl.dump(
                {
                    "U": U,
                    "V": V,
                    "s": s,
                    "json": protmap
                }, svdf)
        U = U[:, :svd_r]
        V = V[:svd_r, :]
        s = s[:svd_r]
        ssqrt = np.sqrt(s)
        Ud = U * ssqrt[None, :]
        Vd = V * ssqrt[:, None]
        return Ud, Vd


def linear_munk(Ua, Va, Ub,
                mapA, mapB, isorank_file, no_landmarks, printOut=True):
    """
    dim T = [svd_dim, svd_dim]
    dim Ub_L = dim Ua_L = [no_landmarks, svd_dim]
    Da = Ua Va
    Ub_L -> Ua_L
    T^* = \min_T \| Ub_L @ T - Ua_L\|_2
    T^* = {Ub_L}^{\dagger} @ Ua_L
    Ub->a = Ub @ T^* = Ub @ {Ub_L}^{\dagger} @ Ua_L
    munk = Ub->a @ Va
    """
    df = pd.read_csv(isorank_file, sep="\t")
    df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: mapA[x])
    df.iloc[:, 1] = df.iloc[:, 1].apply(lambda x: mapB[x])

    matches = df.loc[:no_landmarks, :].values
    matchesA = list(matches[:, 0])
    matchesB = list(matches[:, 1])
    Ua_l = Ua[matchesA, :]
    Ub_l = Ub[matchesB, :]

    Ub_lpinv = compute_pinv(Ub_l)

    Tb_to_a = Ub @ Ub_lpinv @ Ua_l  # Ub -> [nB, 100]
    munk = Tb_to_a @ Va  # [similarity == Unstable]
    munk = munk.T  # to make row = entries corresponding to species A (target)

    loss = Ua_l
    return munk, loss, Ub_l


def compute_pinv(T, epsilon=1e-4):
    U, s, V = svd(T)
    s = s + epsilon
    sinv = 1 / s
    Sinv = np.zeros(T.T.shape)
    np.fill_diagonal(Sinv, sinv)
    Uinv = V.T @ Sinv @ U.T
    return Uinv


def train_model_and_project(Ua, Va, Ub,
                            mapA, mapB, isorank_file,
                            modelfile, 
                            no_landmarks, lr, weight_decay,
                            no_epoch, 
                            munk_only = False):
    """
    Function for model training / projecting the source embeddings to the target 
    species
    Parameters:
    Ua, Va, Ub => Orthogonal matrices obtained from SVD decomposition on the DSD matrices
    mapA, mapB => protein name to index mappings
    isorank_file => one-to-one alignment obtained from ISORANK
    no_landmarks => A copamundial parameter
    lr           => model learning rate
    weight_decay => training weight decay
    munk_only    => set this to true in case you want MUNK results only
    """
    munk, loss, Ub_l = linear_munk(Ua, Va,
                                   Ub, mapA, mapB, 
                                   isorank_file, no_landmarks)
    if munk_only:
        return munk
    pdata = PredictData(Ub)
    predictloader = DataLoader(pdata, shuffle=False, batch_size=10)
    if os.path.exists(modelfile):
        model = torch.load(modelfile, map_location="cpu")
        model.eval()
        with torch.no_grad():
            svdBnls = []
            for j, data in enumerate(predictloader):
                segment = data
                svdBnls.append(model(segment).squeeze(-1).detach().numpy())
            svdBnl = np.concatenate(svdBnls, axis=0)
            modelsim = (svdBnl @ Va).T
            return modelsim
    else:
        data = CuratedData(loss, Ub_l)
        trainloader = DataLoader(data, shuffle=True, batch_size=10)
        loss_fn = nn.MSELoss()
        model = SpeciesTranslationModel(svd_dim=Ua.shape[1])
        model.train()
        optim = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)
        ep = no_epoch
        logging.info(f"Training...")
        for e in range(ep):
            loss = 0
            for i, data in enumerate(trainloader):
                y, x = data
                optim.zero_grad()
                yhat = model(x)
                closs = loss_fn(y, yhat)
                closs.backward()
                optim.step()
                loss += closs.item()
            loss = loss / (i + 1)
            logging.info(f"\t Epoch {e + 1}: Loss : {loss}")
        if modelfile is not None:
            torch.save(model, modelfile)
        model.eval()
        with torch.no_grad():
            svdBnls = []
            for j, data in enumerate(predictloader):
                segment = model(data)
                svdBnls.append(model(segment).squeeze(-1).detach().numpy())
            svdBnl = np.concatenate(svdBnls, axis=0)
            modelsim = (svdBnl @ Va).T
            return modelsim


def get_go_maps(gofile, nmap, gotype):
    """
    If there is go label, return that go label into the set
    else return an empty set
    """
    df = pd.read_csv(gofile, sep="\t")
    df = df.loc[df["type"] == gotype]
    gomaps = df.loc[:, ["GO", "swissprot"]].groupby("swissprot", as_index=False).aggregate(list)
    gomaps = gomaps.values
    go_outs = {}
    all_gos = set()
    for prot, gos in gomaps:
        if prot in nmap:
            all_gos.update(gos)
            go_outs[nmap[prot]] = set(gos)
    for i in range(len(nmap)):
        if i not in go_outs:
            go_outs[i] = {}
    return go_outs, all_gos


def run_copamundial(config):
    DSDA, Aa, nmapA = compute_dsd_dist(config["ppiAfile"], config["dsdAfile"], config["dsd_threshold"])
    DSDB, Ab, nmapB = compute_dsd_dist(config["ppiBfile"], config["dsdBfile"], config["dsd_threshold"])

    SVDAU, SVDAV = compute_svd(config["svdAfile"], DSDA, nmapA, config["svd_r"])
    SVDBU, SVDBV = compute_svd(config["svdBfile"], DSDB, nmapB, config["svd_r"])
    if config["svd_dist_a_b"] is not None and os.path.exists(config["svd_dist_a_b"]):
        logging.info("SVD transformed distances between species A and B already computed")
        with open(config["svd_dist_a_b"], "rb") as svdf:
            package = pkl.load(svdf)
            DISTS = package["SVD-A->B"]
    else:
        if config["compute_isorank"] and not os.path.exists(config["isorankfile"]):
            logging.info("Computing ISORANK")
            compute_isorank_and_save(Aa, Ab, nmapA, nmapB, config["matchfile"], 
                                    config["speciesA"], config["speciesB"], 
                                    config["no_landmarks"], 
                                    config["isorank_alpha"], config["isorankfile"])
            isorank_file = config["isorankfile"]
        elif os.path.exists(config["isorankfile"]):
            isorank_file = config["isorankfile"]
        elif not config["compute_isorank"]:
            isorank_file = config["matchfile"]
        DISTS = train_model_and_project(SVDAU, SVDAV, SVDBU,
                                        nmapA, nmapB, isorank_file, 
                                        config["modelfile"], config["no_landmarks"],
                                        config["lr"], config["weight_decay"], 
                                        config["no_epoch"], config["munk_only"])
        
        if config["svd_dist_a_b"] is not None:
            np.save(config["svd_dist_a_b"], DISTS)

    results = []
    # settings: nameA, nameB, SVD_emb, landmark, gotype, topkacc, dsd/mundo?, kA, kB, 
    settings = [config["speciesA"], config["speciesB"], config["svd_r"], config["no_landmarks"]]

    if config["compute_go_eval"]:
        # Perform evaluations
        kA = config["kA"] 
        kB = config["kB"] 
        go      = config["go"] 
        metric  = config["metric"]
        gomapsA = {}
        gomapsB = {}
        return_val = None
        
        gomapsA[go], golabelsA = get_go_maps(config["goAfile"], nmapA, go)
        gomapsB[go], golabelsB = get_go_maps(config["goBfile"], nmapB, go)
        golabels = golabelsA.union(golabelsB)
        logging.info(f"GO count: {go} ---- {len(golabels)}")
        score = get_scoring(metric, golabels)
        if config["score_dsd"]:
            settings_dsd = settings + [go, metric, "dsd-knn", kA, -1]
            scores, _ = compute_metric(dsd_func(DSDA, k=kA), score, list(range(len(nmapA))), gomapsA[go],
                                        kfold=5)
            logging.info(f"GO: {go}, DSD, k: {kA}, metric: {metric} ===> {np.average(scores):0.3f} +- {np.std(scores):0.3f}")
            settings_dsd += [np.average(scores), np.std(scores)]
            results.append(settings_dsd)
        if config["munk_only"]:
            method = "MUNDO"
        else:
            method = "copamundial"
        settings_copamundial = settings + [go, metric, f"{method}-knn-weight-{config['wB']:0.3f}", kA, kB]
        scores, _ = compute_metric(
            dsd_func_mundo(DSDA, DISTS, gomapsB[go], k=kA, k_other=kB, weight_other=config["wB"]),
            score, list(range(len(nmapA))), gomapsA[go], kfold=5, seed=121)
        settings_copamundial += [np.average(scores), np.std(scores)]
        logging.info(
            f"GO: {go}, {method}, kA: {kA}, kB: {kB}, metric: {metric} ===> {np.average(scores):0.3f} +- {np.std(scores):0.3f}")
        results.append(settings_copamundial)
        columns = ["Species A", "Species B", "SVD embedding", "Landmark no", "GO type", "Scoring metric",
                   "Prediction method",
                   "kA", "kB", "Average score", "Standard deviation"]
        resultsdf = pd.DataFrame(results, columns=columns)
        resultsdf.to_csv(config["output_eval_file"], sep="\t", index=None, mode="a",
                         header=not os.path.exists(config["output_eval_file"]))
        return np.average(scores)
    return


@hydra.main(version_base = None, config_name = "config", config_path = "configs/")
def main(cfg : DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve = True)
    os.makedirs(cfg["outfolder"], exist_ok = True)
    run_copamundial(cfg)


if __name__ == "__main__":
    main()
