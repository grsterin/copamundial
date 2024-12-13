#!/cluster/tufts/cowenlab/.envs/netalign/bin/python
from __future__ import annotations

import argparse

import glidetools.algorithm.dsd as dsd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import svd
import os
from .data import CuratedData, PredictData
from .model import AttentionModel3
from .isorank import isorank, compute_greedy_assignment
from .predict_score import topk_accs, compute_metric, dsd_func, dsd_func_mundo, scoring_fcn
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

class Config:
    def __init__(self, cfile):
        with open(cfile, "r") as cfl:
            config = yaml.safe_load(cfl)
        self.__dict__.update(config)
        self.ppiAfile = f"{self.netfolder}/{self.speciesA}.s.tsv"
        self.ppiBfile = f"{self.netfolder}/{self.speciesB}.s.tsv"
        self.dsdAfile = f"{self.tempfolder}/{self.speciesA}.dsd.pkl"
        self.dsdBfile = f"{self.tempfolder}/{self.speciesB}.dsd.pkl"
        self.svdAfile = f"{self.tempfolder}/{self.speciesA}.svd.pkl"
        self.svdBfile = f"{self.tempfolder}/{self.speciesB}.svd.pkl"
        self.goAfile = f"{self.gofolder}/{self.speciesA}.output.mapping.gaf"
        self.goBfile = f"{self.gofolder}/{self.speciesB}.output.mapping.gaf"
        self.svd_dist_a_b = f"{self.tempfolder}/t_svd_{self.svd_r}_{self.speciesA}-{self.speciesB}.svd.pkl"
        self.isorankfile = f"{self.tempfolder}/{self.speciesA}-{self.speciesB}_alpha_{self.isorank_alpha}_N_{self.no_landmarks}.isorank.tsv"
        self.matchfile = f"{self.netfolder}/{self.speciesA}-{self.speciesB}.tsv"
        if not os.path.exists(self.matchfile):
            self.matchfile = f"{self.netfolder}/{self.speciesB}-{self.speciesA}.tsv"
        self.modelfile = f"{self.tempfolder}/{self.speciesA}-{self.speciesB}_lr_{self.lr}_ep_{self.no_epoch}.model.pt"
        return


def compute_pairs(df, nmapA, nmapB, config):
    df = df.loc[:, [config.speciesA, config.speciesB, "score"]]

    df[config.speciesA] = df[config.speciesA].apply(lambda x: nmapA[x])
    df[config.speciesB] = df[config.speciesB].apply(lambda x: nmapB[x])

    m, n = len(nmapA), len(nmapB)
    E = np.zeros((m, n))

    for p, q, v in df.values:
        E[int(p + 0.25), int(q + 0.25)] = v
    return E


def get_scoring(metric, all_go_labels=None, **kwargs):
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


def compute_isorank_and_save(Aa, Ab, mapA, mapB, config):
    rmapA = {v: k for k, v in mapA.items()}
    rmapB = {v: k for k, v in mapB.items()}

    pdmatch = pd.read_csv(config.matchfile, sep="\t")
    pdmatch = pdmatch.loc[
              pdmatch[config.speciesA].apply(lambda x: x in mapA) & pdmatch[config.speciesB].apply(lambda x: x in mapB),
              :]

    print(f"[!!] \tSize of the matchfile: {len(pdmatch)}")

    E = compute_pairs(pdmatch, mapA,
                      mapB, config)

    R0 = isorank(Aa, Ab, E, config.isorank_alpha, maxiter=-1)
    align = compute_greedy_assignment(R0, config.no_landmarks)
    aligndf = pd.DataFrame(align, columns=[config.speciesA, config.speciesB])
    aligndf.iloc[:, 0] = aligndf.iloc[:, 0].apply(lambda x: rmapA[x])
    aligndf.iloc[:, 1] = aligndf.iloc[:, 1].apply(lambda x: rmapB[x])
    aligndf.to_csv(config.isorankfile, sep="\t", index=None)
    return


def compute_dsd_dist(config, ppifile, dsdfile):
    if dsdfile is not None and os.path.exists(dsdfile):
        print(f"DSD file already computed <- {dsdfile}")
        with open(dsdfile, "rb") as cfdsd:
            package = pkl.load(cfdsd)
            DSDdist = package["dsd_dist"]
            protmap = package["json"]
            Asub = package["A"]
        if config.dsd_threshold > 0:
            DSDdist = np.where(DSDdist > config.dsd_threshold,
                               config.dsd_threshold,
                               DSDdist)
        return DSDdist, Asub, protmap
    else:
        print(f"Computing DSD file -> {dsdfile}")
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
        if config.dsd_threshold > 0:
            DSDdist = np.where(DSDdist > config.dsd_threshold,
                               config.dsd_threshold, DSDdist)
        with open(dsdfile, "wb") as cfdsd:
            pkl.dump({
                "A": Asub,
                "dsd_dist": DSDdist,
                "dsd_emb": DSDemb,
                "json": protmap
            }, cfdsd)
        return DSDdist, Asub, protmap


# SVD FILE
def compute_svd(config, svdfile, dsddist, protmap):
    if os.path.exists(svdfile):
        print(f"SVD files already computed <- {svdfile}")
        with open(svdfile, "rb") as svdf:
            package = pkl.load(svdf)
            U = package["U"]
            V = package["V"]
            s = package["s"]
        U = U[:, :config.svd_r]
        V = V[:config.svd_r, :]
        s = s[:config.svd_r]
        ssqrt = np.sqrt(s)
        Ud = U * ssqrt[None, :]  # [n, svd]
        Vd = V * ssqrt[:, None]  # [svd, n]
        return Ud, Vd
    else:
        print(f"Computing the SVD file -> {svdfile}")
        U, s, V = svd(dsddist)
        with open(svdfile, "wb") as svdf:
            pkl.dump(
                {
                    "U": U,
                    "V": V,
                    "s": s,
                    "json": protmap
                }, svdf)
        U = U[:, :config.svd_r]
        V = V[:config.svd_r, :]
        s = s[:config.svd_r]
        ssqrt = np.sqrt(s)
        Ud = U * ssqrt[None, :]
        Vd = V * ssqrt[:, None]
        return Ud, Vd


def linear_munk(config, Ua, Va, Ub,
                mapA, mapB, isorank_file, printOut=True):
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

    matches = df.loc[:config.no_landmarks, :].values
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
    Uupdated = U @ np.diag(s) @ V
    Uinv = V.T @ np.diag(sinv) @ U.T
    print(Uupdated @ Uinv, np.linalg.cond(Uinv))
    return Uinv


def train_model_and_project(config, Ua, Va, Ub,
                            mapA, mapB, isorank_file):
    munk, loss, Ub_l = linear_munk(config, Ua, Va,
                                   Ub, mapA, mapB, isorank_file)
    if hasattr(config, "munkonly") and config.munkonly:
        return munk
    pdata = PredictData(Ub)
    predictloader = DataLoader(pdata, shuffle=False, batch_size=10)
    if os.path.exists(config.modelfile):
        model = torch.load(config.modelfile, map_location="cpu")
        model.eval()
        with torch.no_grad():
            svdBnls = []
            for j, data in enumerate(predictloader):
                segment = data
                svdBnls.append(model(segment).squeeze(-1).detach().numpy())
            svdBnl = np.concatenate(svdBnls, axis=0)
            modelsim = (svdBnl @ Va).T
            print(f"modelsim: {modelsim}")
            return modelsim
    else:
        data = CuratedData(loss, Ub_l)
        trainloader = DataLoader(data, shuffle=True, batch_size=10)
        loss_fn = nn.MSELoss()
        model = AttentionModel3(svd_dim=Ua.shape[1])
        model.train()
        optim = torch.optim.Adam(model.parameters(),
                                 lr=config.lr,
                                 weight_decay=config.weight_decay)
        ep = config.no_epoch
        print(f"Training...")
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
            print(f"\t Epoch {e + 1}: Loss : {loss}")
        if config.modelfile is not None:
            torch.save(model, config.modelfile)
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
    DSDA, Aa, nmapA = compute_dsd_dist(config, config.ppiAfile, config.dsdAfile)
    DSDB, Ab, nmapB = compute_dsd_dist(config, config.ppiBfile, config.dsdBfile)

    SVDAU, SVDAV = compute_svd(config, config.svdAfile, DSDA, nmapA)
    SVDBU, SVDBV = compute_svd(config, config.svdBfile, DSDB, nmapB)
    if config.svd_dist_a_b is not None and os.path.exists(config.svd_dist_a_b):
        print("SVD transformed distances between species A and B already computed")
        with open(config.svd_dist_a_b, "rb") as svdf:
            package = pkl.load(svdf)
            DISTS = package["SVD-A->B"]
    else:
        if config.compute_isorank and not os.path.exists(config.isorankfile):
            print("Computing ISORANK")
            compute_isorank_and_save(Aa, Ab, nmapA, nmapB, config)
            isorank_file = config.isorankfile
        elif os.path.exists(config.isorankfile):
            isorank_file = config.isorankfile
        elif not config.compute_isorank:
            isorank_file = config.matchfile
        DISTS = train_model_and_project(config,
                                        SVDAU, SVDAV, SVDBU,
                                        nmapA, nmapB, isorank_file)
        if config.svd_dist_a_b is not None:
            np.save(config.svd_dist_a_b, DISTS)

    results = []
    # settings: nameA, nameB, SVD_emb, landmark, gotype, topkacc, dsd/mundo?, kA, kB, 
    settings = [config.speciesA, config.speciesB, config.svd_r, config.no_landmarks]

    if config.compute_go_eval:
        # Perform evaluations
        # go_
        kAs = [int(k) for k in config.kA]
        kBs = [int(k) for k in config.kB]
        gos = config.gos
        gomapsA = {}
        gomapsB = {}
        for go in gos:
            gomapsA[go], golabelsA = get_go_maps(config.goAfile, nmapA, go)
            gomapsB[go], golabelsB = get_go_maps(config.goBfile, nmapB, go)
            golabels = golabelsA.union(golabelsB)
            print(f"GO count: {go} ---- {len(golabels)}")
            for metric in config.metrics:
                score = get_scoring(metric, golabels)
                for kA in kAs:
                    if config.score_dsd:
                        settings_dsd = settings + [go, metric, "dsd-knn", kA, -1]
                        scores, _ = compute_metric(dsd_func(DSDA, k=kA), score, list(range(len(nmapA))), gomapsA[go],
                                                   kfold=5)
                        print(f"GO: {go}, DSD, k: {kA} ===> {np.average(scores):0.3f} +- {np.std(scores):0.3f}")
                        settings_dsd += [np.average(scores), np.std(scores)]
                        results.append(settings_dsd)
                    for kB in kBs:
                        settings_mundo = settings + [go, metric, f"mundo4-knn-weight-{config.wB:0.3f}", kA, kB]
                        scores, _ = compute_metric(
                            dsd_func_mundo(DSDA, DISTS, gomapsB[go], k=kA, k_other=kB, weight_other=config.wB),
                            score, list(range(len(nmapA))), gomapsA[go], kfold=5, seed=121)
                        settings_mundo += [np.average(scores), np.std(scores)]

                        print(
                            f"GO: {go}, MUNDO4, kA: {kA}, kB: {kB} ===> {np.average(scores):0.3f} +- {np.std(scores):0.3f}")
                        results.append(settings_mundo)
        columns = ["Species A", "Species B", "SVD embedding", "Landmark no", "GO type", "Scoring metric",
                   "Prediction method",
                   "kA", "kB", "Average score", "Standard deviation"]
        resultsdf = pd.DataFrame(results, columns=columns)
        resultsdf.to_csv(config.output_eval_file, sep="\t", index=None, mode="a",
                         header=not os.path.exists(config.output_eval_file))
    return

class CitationAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super(CitationAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        from . import __citation__

        print(__citation__)
        setattr(namespace, self.dest, values)
        sys.exit(0)

def main():
    from . import __version__

    parser = argparse.ArgumentParser()

    parser.add_argument("-v", "--version", action="version", version="Copamundial " + __version__)
    parser.add_argument(
        "-c",
        "--citation",
        action=CitationAction,
        nargs=0,
        help="show program's citation and exit"
    )
    parser.add_argument("--config", help="Config YAML file", required=True)
    args = parser.parse_args()

    config = Config(args.config)
    run_copamundial(config)
