from __future__ import annotations
import sys
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


def compute_ppi(ppifile):
    """
    get the PPI adjacency matrix
    """
    ppdf = pd.read_csv(ppifile, sep = "\t", header = None)
    Gpp  = nx.from_pandas_edgelist(ppdf, source = 0, 
                                  target = 1)
    ccs  = max(nx.connected_components(Gpp), key=len)
    Gpps = Gpp.subgraph(ccs)
    Asub = nx.to_numpy_array(Gpps)
    protmap = {k: i for i, k in enumerate(list(Gpps.nodes))}
    return Asub, protmap


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
        Gpp  = nx.from_pandas_edgelist(ppdf, source=0,
                                      target=1)
        ccs  = max(nx.connected_components(Gpp), key=len)
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

EPSILON = 1e-4

def run_isorank_fp(config):
    DSDA, Aa, nmapA = compute_dsd_dist(config["ppiAfile"], config["dsdAfile"], config["dsd_threshold"])
    Ab, nmapB       = compute_ppi(config["ppiBfile"])
    match_file      = config["matchfile"]
    speciesA, speciesB = config["speciesA"], config["speciesB"]
    isorank_alpha   = config["isorank_alpha"]
    logging.info("Computing ISORANK alignments")
    pdmatch         = pd.read_csv(match_file, sep = "\t")

    pdmatch = pdmatch.loc[
              pdmatch[speciesA].apply(lambda x: x in nmapA) & pdmatch[speciesB].apply(lambda x: x in nmapB),
              :]

    E               = compute_pairs(pdmatch, nmapA,
                                    nmapB, speciesA, speciesB)
    R0              = isorank(Aa, Ab, E, isorank_alpha, maxiter=-1)
    DISTS           = 1 / (R0 + EPSILON)    
    
    results = []
    # settings: nameA, nameB, SVD_emb, landmark, gotype, topkacc, dsd/mundo?, kA, kB, 
    settings = [config["speciesA"], config["speciesB"], config["no_landmarks"]]

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
            method = "ISORANK"
        settings_copamundial = settings + [go, metric, f"{method}-knn-weight-{config['wB']:0.3f}", kA, kB]
        scores, _ = compute_metric(
            dsd_func_mundo(DSDA, DISTS, gomapsB[go], k=kA, k_other=kB, weight_other=config["wB"]),
            score, list(range(len(nmapA))), gomapsA[go], kfold=5, seed=121)
        settings_copamundial += [np.average(scores), np.std(scores)]
        logging.info(
            f"GO: {go}, {method}, kA: {kA}, kB: {kB}, metric: {metric} ===> {np.average(scores):0.3f} +- {np.std(scores):0.3f}")
        results.append(settings_copamundial)
        columns = ["Species A", "Species B", "Landmark no", "GO type", "Scoring metric",
                   "Prediction method",
                   "kA", "kB", "Average score", "Standard deviation"]
        resultsdf = pd.DataFrame(results, columns=columns)
        resultsdf.to_csv(config["output_eval_file"], sep="\t", index=None, mode="a",
                         header=not os.path.exists(config["output_eval_file"]))
        return np.average(scores) #return the average score
    return 


@hydra.main(version_base = None, config_name = "config", config_path = "configs/")
def main(cfg : DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve = True)
    os.makedirs(cfg["outfolder"], exist_ok = True)
    res = run_isorank_fp(cfg)
    return res

if __name__ == "__main__":
    main()
