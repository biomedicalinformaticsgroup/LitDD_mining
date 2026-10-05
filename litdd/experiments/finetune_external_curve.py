#!/usr/bin/env python3
"""Learning curve of held-out external recall over the number of external gene buckets trained on.

Reads ``ds_bert_train`` and ``ds_test`` from ``--data_dir``, an external truth CSV
(``--external_csv``: ``pmid, tiab, gene, source``), an annotation worksheet (``--aug_csv``) and
a random PubMed sample (``--random_csv``: ``pmid, tiab``).

Every external paper's genes are bucketed by ``gene_fold(gene, 10)``. Papers whose genes all
fall in bucket 0 form the held-out evaluation set; papers with any bucket-0 gene and others are
dropped; the rest are trainable at level ``max(bucket)``. For each level ``k`` in ``--levels``
the base set (``ds_bert_train`` plus the confirmed worksheet rows) is extended with the
trainable papers of levels 1..k as positives, a model is trained and evaluated on ``ds_test``
(precision, recall, F1), on the held-out set (recall per source and overall, threshold 0.5) and
on the random sample (positive rate and inference throughput). Writes one row per level to
``--out_csv`` and the held-out probabilities per level under ``--dump_scores``; with
``--save_dir`` the model of the last level is saved there.
"""
from __future__ import annotations

import argparse
import gc
import os
import time

import pandas as pd

from litdd.training.screen_common import as_positive_ds, aug_ds, gene_fold, score_proba, train_and_eval

MODEL = "thomas-sounack/BioClinical-ModernBERT-large"
HP = dict(epochs=5, bs=32, lr=1.736e-5, wd=0.3)
MAX_LENGTH = 512
SOURCES = ["premined", "hpoa", "clingen"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True, help="directory with ds_bert_train and ds_test")
    ap.add_argument("--external_csv", required=True, help="external truth CSV (pmid, tiab, gene, source)")
    ap.add_argument("--aug_csv", required=True, help="annotation worksheet with a confirm_positive column")
    ap.add_argument("--random_csv", required=True, help="random PubMed sample (pmid, tiab)")
    ap.add_argument("--levels", default="0,1,3,6,9", help="cumulative gene-bucket levels (of 9) to add")
    ap.add_argument("--out_csv", required=True, help="results CSV, one row per level")
    ap.add_argument("--dump_scores", required=True, help="directory for held-out probability CSVs")
    ap.add_argument("--save_dir", default=None,
                    help="if set, save the model and tokenizer of the last level here")
    ap.add_argument("--lr", type=float, default=None, help="override the learning rate")
    ap.add_argument("--wd", type=float, default=None, help="override the weight decay")
    ap.add_argument("--epochs", type=int, default=None, help="override the number of epochs")
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.lr is not None:
        HP["lr"] = args.lr
    if args.wd is not None:
        HP["wd"] = args.wd
    if args.epochs is not None:
        HP["epochs"] = args.epochs
    print(f"HP: {HP}")
    from datasets import concatenate_datasets, load_from_disk
    ds_train = load_from_disk(f"{args.data_dir}/ds_bert_train")
    ds_test = load_from_disk(f"{args.data_dir}/ds_test")

    ext = pd.read_csv(args.external_csv, dtype=str).drop_duplicates("pmid").copy()
    # Per-paper gene buckets; held-out papers have every gene in bucket 0.
    buckets = ext.groupby("pmid")["gene"].apply(lambda gs: {gene_fold(g, 10) for g in gs})
    ext["gbuckets"] = ext["pmid"].map(buckets)
    ext["is_heldout"] = ext["gbuckets"].map(lambda b: b == {0})
    ext["is_mixed"] = ext["gbuckets"].map(lambda b: (0 in b) and b != {0})
    ext["level"] = ext["gbuckets"].map(lambda b: max(b) if 0 not in b else None)
    heldout = ext[ext["is_heldout"]].reset_index(drop=True)
    trainable = ext[(~ext["is_heldout"]) & (~ext["is_mixed"])]
    levels = [int(x) for x in args.levels.split(",")]
    print(f"held-out {len(heldout)} ({heldout['source'].value_counts().to_dict()}) | "
          f"trainable pool {len(trainable)} | levels {levels}")
    for k in levels:
        n = (trainable["level"] <= k).sum() if k > 0 else 0
        print(f"  level {k}/9 -> +{n} external train papers")
    if args.dry_run:
        print("dry_run OK")
        return

    import torch
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    base = concatenate_datasets([ds_train, aug_ds(args.aug_csv, ds_train.features)[0]])
    rnd = pd.read_csv(args.random_csv, dtype=str).fillna("")
    os.makedirs(args.dump_scores, exist_ok=True)
    rows = []
    for k in levels:
        add = trainable[trainable["level"] <= k] if k > 0 else trainable.iloc[:0]
        tr_ds = base if k == 0 else concatenate_datasets([base, as_positive_ds(add, ds_train.features)])
        print(f"\n=== level {k}/9: +{len(add)} external, train {tr_ds.num_rows} ===", flush=True)
        model, f1 = train_and_eval(tr_ds, ds_test, tokenizer, model_name=MODEL, hp=HP,
                                   out_dir=f"./_curve_{k}", logging_steps=300, max_length=MAX_LENGTH,
                                   bf16=torch.cuda.is_available())
        ho_p = score_proba(model, tokenizer, heldout["tiab"], MAX_LENGTH)  # also warms up the GPU
        t0 = time.time()
        rnd_p = score_proba(model, tokenizer, rnd["tiab"], MAX_LENGTH)
        aps = round(len(rnd) / (time.time() - t0), 1)  # tokenisation plus forward pass
        print(f"  inference throughput: {aps} abstracts/sec ({len(rnd)} abstracts)")
        row = {"level": k, "n_external": len(add), **f1, "random_fpr_pct": round(100 * (rnd_p >= 0.5).mean(), 2),
               "infer_abstracts_per_sec": aps}
        if args.save_dir and k == levels[-1]:
            model.save_pretrained(args.save_dir)
            tokenizer.save_pretrained(args.save_dir)
            print(f"  saved checkpoint -> {args.save_dir}")
        for s in SOURCES:
            m = (heldout["source"] == s).values
            row[f"heldout_recall_{s}_pct"] = round(100 * (ho_p[m] >= 0.5).mean(), 1) if m.any() else None
        row["heldout_recall_all_pct"] = round(100 * (ho_p >= 0.5).mean(), 1)
        rows.append(row)
        print(row)
        pd.DataFrame({"source": heldout["source"].values, "proba": ho_p}).to_csv(
            f"{args.dump_scores}/level{k}_heldout.csv", index=False)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print("\n=== external-recall learning curve ===")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
