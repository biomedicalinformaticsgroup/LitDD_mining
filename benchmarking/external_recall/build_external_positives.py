#!/usr/bin/env python3
"""Build the external-set positive training pool for the high-recall screen experiment (R3.3).

Takes the external truth papers (premined DDG2P `publications` / ClinGen / HPOA), attaches
title+abstract, keeps only gene-present papers (G2P gene symbol OR previous/alias symbol OR
NCBI full gene name in the TIAB — the design exclusion for no-molecular-confirmation papers),
excludes anything already in the annotated train/test set, and assigns each PAPER a single
gene-fold split so training and evaluation never share a gene:
  fold(gene) = md5(gene) % 5 == 0 -> held-out, else train
  split = 'train'   if every truth-gene of the paper is train-fold   (-> add to training)
          'heldout' if every truth-gene is held-out-fold             (-> recall eval, unseen genes)
          'mixed'   otherwise                                        (-> dropped, avoids leakage)

A recall lift on the held-out-fold papers after adding the train-fold papers to training is
generalisation of the register to unseen genes, not memorisation. Output feeds
train_test/finetune_external_recall.py.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import re

import pandas as pd


def fold(gene: str) -> str:
    return "heldout" if int(hashlib.md5(gene.encode()).hexdigest(), 16) % 5 == 0 else "train"


def gene_fullnames(path: str) -> dict[str, str]:
    out = {}
    with gzip.open(path, "rt") as fh:
        h = {c: i for i, c in enumerate(fh.readline().lstrip("#").rstrip().split("\t"))}
        for ln in fh:
            p = ln.rstrip("\n").split("\t")
            if p[h["description"]] not in ("", "-"):
                out[p[h["Symbol"]]] = p[h["description"]]
    return out


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truthsets", default="revision/external_recall/truthsets.csv")
    ap.add_argument("--tiab_csv", default="revision/external_recall/evagg_external_eval.csv",
                    help="pmid,tiab for the truth papers")
    ap.add_argument("--ddg2p", required=True)
    ap.add_argument("--annotated", required=True, help="annotated_tiab.csv (train/test PMIDs to exclude)")
    ap.add_argument("--gene_info", required=True)
    ap.add_argument("--out", default="revision/external_recall/external_positives.csv")
    return ap.parse_args()


def main():
    args = parse_args()
    dd = pd.read_csv(args.ddg2p, dtype=str).fillna("")
    dd.columns = [c.strip() for c in dd.columns]
    g2gene = dict(zip(dd["g2p id"], dd["gene symbol"].str.strip()))
    g2prev = dict(zip(dd["g2p id"], dd["previous gene symbols"]))
    names = gene_fullnames(args.gene_info)
    tiab = pd.read_csv(args.tiab_csv, dtype=str).drop_duplicates("pmid").set_index("pmid")["tiab"].to_dict()
    ann = set(pd.read_csv(args.annotated, dtype=str, usecols=["pmid"])["pmid"].astype(str).str.strip())
    t = pd.read_csv(args.truthsets, dtype=str)

    def present(gene, prev, txt):
        txt = txt.lower()
        forms = [gene] + [x.strip() for x in prev.replace(";", ",").split(",") if x.strip()]
        if names.get(gene):
            forms.append(names[gene])
        return any(re.search(r"\b" + re.escape(f.lower()) + r"\b", txt) for f in forms if len(f) > 2)

    rows = []
    for r in t.itertuples():
        pm = str(r.pmid); gid = r.key; gene = g2gene.get(gid, "")
        if pm not in tiab or not gene or pm in ann:
            continue
        if not present(gene, g2prev.get(gid, ""), tiab[pm]):
            continue
        rows.append({"pmid": pm, "g2p_id": gid, "gene": gene, "source": r.source,
                     "fold": fold(gene), "tiab": tiab[pm]})
    ext = pd.DataFrame(rows)
    g = ext.groupby("pmid")["fold"].agg(
        lambda s: "heldout" if set(s) == {"heldout"} else ("train" if set(s) == {"train"} else "mixed"))
    ext["split"] = ext["pmid"].map(g)
    ext.to_csv(args.out, index=False)
    paper = ext.drop_duplicates("pmid")
    print(f"gene-present external truth papers (not in annotated set): {paper['pmid'].nunique()}")
    print(f"by split: {paper['split'].value_counts().to_dict()}")
    print(f"train_add by source:    {paper[paper.split == 'train']['source'].value_counts().to_dict()}")
    print(f"eval_heldout by source: {paper[paper.split == 'heldout']['source'].value_counts().to_dict()}")
    print(f"-> {args.out} ({len(ext)} rows)")


if __name__ == "__main__":
    main()
