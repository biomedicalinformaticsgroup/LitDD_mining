#!/usr/bin/env python3
"""Measure the effect of adding external curated positives to the screen's training set.

Reads ``ds_bert_train`` and ``ds_test`` from ``--data_dir``, an augmentation worksheet
(``--aug_csv``; ``confirm_positive`` decides the label), an external truth CSV
(``--external_csv``: ``pmid, tiab, g2p_id, source, fold, split``) and a random PubMed sample
(``--random_csv``: ``pmid, tiab``). External papers already in the worksheet are dropped so no
paper is counted twice.

Two models are trained with the same hyperparameters:

  augmented                ds_bert_train plus the confirmed worksheet rows
  augmented_plus_external  the same plus the external rows with ``split == train`` as positives

Each model is evaluated on ``ds_test`` (precision, recall, F1), on the external rows with
``split == heldout`` (recall per source and overall, threshold 0.5) and on the random sample
(positive rate). One row per variant is written to ``--out_csv``; the per-paper probabilities
for the held-out and random sets are written under ``--dump_scores`` for threshold sweeps.
"""
from __future__ import annotations

import argparse
import gc
import os

import pandas as pd

from litdd.training.screen_common import as_positive_ds, aug_ds, score_proba, train_and_eval

MODEL = "thomas-sounack/BioClinical-ModernBERT-large"
HP = dict(epochs=5, bs=32, lr=1.736e-5, wd=0.3)
MAX_LENGTH = 512
SOURCES = ["premined", "hpoa", "clingen"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True, help="directory with ds_bert_train and ds_test")
    ap.add_argument("--external_csv", required=True, help="external_positives.csv (pmid,tiab,g2p_id,source,fold,split)")
    ap.add_argument("--aug_csv", required=True, help="augmentation worksheet with a confirm_positive column")
    ap.add_argument("--random_csv", required=True, help="random PubMed sample (pmid, tiab)")
    ap.add_argument("--out_csv", required=True, help="results CSV, one row per variant")
    ap.add_argument("--dump_scores", required=True, help="directory for per-paper probability CSVs")
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    from datasets import concatenate_datasets, load_from_disk
    ds_train = load_from_disk(f"{args.data_dir}/ds_bert_train")
    ds_test = load_from_disk(f"{args.data_dir}/ds_test")

    aug, aug_pmids = aug_ds(args.aug_csv, ds_train.features)
    base = concatenate_datasets([ds_train, aug])
    ext = pd.read_csv(args.external_csv, dtype=str).drop_duplicates("pmid")
    ext = ext[~ext["pmid"].astype(str).isin(aug_pmids)]
    train_add = ext[ext["split"] == "train"]
    heldout = ext[ext["split"] == "heldout"].reset_index(drop=True)
    rnd = pd.read_csv(args.random_csv, dtype=str).fillna("")
    print(f"base = ds_bert_train {ds_train.num_rows} + augmentation {aug.num_rows} = {base.num_rows} "
          f"(pos {sum(base['label'])}) | external train-add {len(train_add)} | "
          f"held-out eval {len(heldout)} ({heldout['source'].value_counts().to_dict()}) | random {len(rnd)}")
    if args.dry_run:
        print("dry_run OK")
        return

    import torch
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    ext_pos = as_positive_ds(train_add, ds_train.features)
    runs = [("augmented", base), ("augmented_plus_external", concatenate_datasets([base, ext_pos]))]
    os.makedirs(args.dump_scores, exist_ok=True)
    rows = []
    for label, tr_ds in runs:
        print(f"\n=== variant={label}: train {tr_ds.num_rows} ===", flush=True)
        model, f1 = train_and_eval(tr_ds, ds_test, tokenizer, model_name=MODEL, hp=HP,
                                   out_dir=f"./_ext_{label}", logging_steps=200,
                                   max_length=MAX_LENGTH, bf16=torch.cuda.is_available())
        ho_p = score_proba(model, tokenizer, heldout["tiab"], MAX_LENGTH)
        rnd_p = score_proba(model, tokenizer, rnd["tiab"], MAX_LENGTH)
        row = {"variant": label, **f1, "random_fpr_pct": round(100 * (rnd_p >= 0.5).mean(), 2)}
        for s in SOURCES:
            mask = (heldout["source"] == s).values
            row[f"heldout_recall_{s}_pct"] = round(100 * (ho_p[mask] >= 0.5).mean(), 1) if mask.any() else None
        row["heldout_recall_all_pct"] = round(100 * (ho_p >= 0.5).mean(), 1)
        rows.append(row)
        print(row)
        pd.DataFrame({"source": heldout["source"].values, "proba": ho_p}).to_csv(
            f"{args.dump_scores}/{label}_heldout.csv", index=False)
        pd.DataFrame({"proba": rnd_p}).to_csv(f"{args.dump_scores}/{label}_random.csv", index=False)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print("\n=== external-recall results (held-out split) ===")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
