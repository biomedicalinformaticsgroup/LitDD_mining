#!/usr/bin/env python3
"""Train the screen under several random seeds and report the spread of its metrics.

Reads a training dataset (HuggingFace ``save_to_disk`` directory), the ``ds_test`` split, an
external truth CSV (``pmid, tiab, gene, source``) and a random PubMed sample CSV (``pmid,
tiab``). For each seed it fine-tunes BioClinical-ModernBERT-large with the CV-selected
hyperparameters, evaluates precision, recall and F1 on ``ds_test``, scores the external papers
whose genes all fall in held-out fold 0 of 10 (recall per source and overall) and the random
sample (positive rate), and appends one row per seed to ``--out_csv``.

With ``--save_dir`` every seed's model and tokenizer are saved under ``<save_dir>/seed_<n>``.
The released checkpoint is seed 44. Prints the per-seed rows and the mean and standard
deviation over seeds.
"""
from __future__ import annotations

import argparse
import gc
import os

import pandas as pd

from litdd.training.screen_common import gene_fold, score_proba, train_and_eval

# ModernBERT context length; training and inference use the same cap.
MAX_LENGTH = 8192
MODEL = "thomas-sounack/BioClinical-ModernBERT-large"
HP = dict(epochs=5, bs=32, lr=3e-5, wd=0.1)  # selected by cv_hp_search_bert.py
SOURCES = ["premined", "hpoa", "clingen"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train_ds_dir", required=True, help="training dataset (save_to_disk directory)")
    ap.add_argument("--test_ds_dir", required=True, help="directory containing ds_test")
    ap.add_argument("--external_csv", required=True, help="external truth CSV (pmid, tiab, gene, source)")
    ap.add_argument("--random_csv", required=True, help="random PubMed sample (pmid, tiab)")
    ap.add_argument("--seeds", default="42,43,44", help="comma-separated seeds")
    ap.add_argument("--save_dir", default=None,
                    help="if set, save every seed's model and tokenizer under <save_dir>/seed_<n>")
    ap.add_argument("--out_csv", required=True, help="per-seed results CSV")
    ap.add_argument("--attn_implementation", default=None,
                    help="attention implementation passed to from_pretrained (e.g. sdpa)")
    ap.add_argument("--train_bs", type=int, default=HP["bs"],
                    help="per-device train batch size; with --grad_accum keeps the effective batch of 32")
    ap.add_argument("--grad_accum", type=int, default=1, help="gradient-accumulation steps")
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    from datasets import load_from_disk
    ds_train = load_from_disk(args.train_ds_dir)
    ds_test = load_from_disk(f"{args.test_ds_dir}/ds_test")
    ext = pd.read_csv(args.external_csv, dtype=str).drop_duplicates("pmid").copy()
    # Held-out papers: every gene of the paper falls in fold 0 of 10.
    folds = ext.groupby("pmid")["gene"].apply(lambda gs: {gene_fold(g, 10) for g in gs})
    ho = ext[ext["pmid"].map(folds).map(lambda s: s == {0})].reset_index(drop=True)
    rnd = pd.read_csv(args.random_csv, dtype=str).fillna("")
    seeds = [int(s) for s in args.seeds.split(",")]
    hp = dict(HP, bs=args.train_bs, grad_accum=args.grad_accum)
    print(f"[INFO] hyperparameters {hp} (effective batch {hp['bs'] * hp['grad_accum']})", flush=True)
    print(f"train {ds_train.num_rows} | test {ds_test.num_rows} | held-out {len(ho)} "
          f"({ho['source'].value_counts().to_dict()}) | seeds {seeds}")
    if args.dry_run:
        print("dry_run OK")
        return

    import torch
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    rows = []
    for seed in seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        model, f1 = train_and_eval(ds_train, ds_test, tokenizer, model_name=MODEL, hp=hp,
                                   out_dir=f"./_seed_{seed}", logging_steps=300,
                                   max_length=MAX_LENGTH, bf16=False, seed=seed,
                                   attn_implementation=args.attn_implementation)
        ho_p = score_proba(model, tokenizer, ho["tiab"], MAX_LENGTH)
        rnd_p = score_proba(model, tokenizer, rnd["tiab"], MAX_LENGTH)
        row = {"seed": seed, **f1, "random_fpr_pct": round(100 * (rnd_p >= 0.5).mean(), 2)}
        for s in SOURCES:
            m = (ho["source"] == s).values
            row[f"heldout_recall_{s}_pct"] = round(100 * (ho_p[m] >= 0.5).mean(), 1)
        row["heldout_recall_all_pct"] = round(100 * (ho_p >= 0.5).mean(), 1)
        rows.append(row)
        print(row)
        if args.save_dir:
            d = os.path.join(args.save_dir, f"seed_{seed}")
            model.save_pretrained(d)
            tokenizer.save_pretrained(d)
            print(f"[INFO] saved seed {seed} -> {d}", flush=True)
            tokenizer.save_pretrained(args.save_dir)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    num = out.select_dtypes("number").drop(columns=["seed"])
    print("\n=== per seed ===")
    print(out.to_string(index=False))
    print("\n=== mean +/- std over seeds ===")
    for c in num.columns:
        print(f"  {c}: {num[c].mean():.2f} +/- {num[c].std():.2f}")


if __name__ == "__main__":
    main()
