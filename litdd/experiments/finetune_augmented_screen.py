#!/usr/bin/env python3
"""Train the screen with and without the confirmed augmentation rows and measure miss recovery.

Reads ``ds_bert_train`` and ``ds_test`` from ``--data_dir``, an annotation worksheet
(``--aug_csv``), the gene-present screen misses (``--misses_csv``: ``pmid, title, abstract,
g2p``) and the G2P DD CSV (``--ddg2p``, to map each miss's G2P id to its gene). Optionally a
random PubMed sample (``--random_csv``: ``pmid, tiab``).

Trains a baseline on ``ds_bert_train`` and an augmented model on ``ds_bert_train`` plus the
confirmed worksheet rows, or, with ``--aug_sizes``, one model per number of augmentation
positives (a learning curve). Each model is evaluated on ``ds_test`` (precision, recall, F1),
on the misses (share predicted positive, overall and split by whether the gene falls in
held-out fold 0 of 5) and, when given, on the random sample (positive rate). Writes one row
per model to ``--out_csv``; with ``--dump_scores`` also writes the positive-class probability
of every test, miss and random example per model.
"""
from __future__ import annotations

import argparse
import gc
import os

import pandas as pd

from litdd.training.screen_common import aug_ds, fold_name, score_argmax, score_proba, train_and_eval

MODEL = "thomas-sounack/BioClinical-ModernBERT-large"
HP = dict(epochs=5, bs=32, eval_bs=32, lr=1.736e-5, wd=0.3)
MAX_LENGTH = 512


def load_misses(csv_path: str, ddg2p: str) -> pd.DataFrame:
    """Misses with ``tiab`` (title and abstract) and ``fold`` (held-out or train, by gene)."""
    m = pd.read_csv(csv_path, dtype=str).fillna("").drop_duplicates("pmid")
    dd = pd.read_csv(ddg2p, dtype=str).fillna("")
    dd.columns = [c.strip() for c in dd.columns]
    g2gene = dict(zip(dd["g2p id"], dd["gene symbol"].str.strip()))
    m["tiab"] = (m["title"] + " " + m["abstract"]).str.strip()
    m["fold"] = m["g2p"].map(lambda g: fold_name(g2gene.get(g, ""), 5))
    return m


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True, help="directory with ds_bert_train and ds_test")
    ap.add_argument("--aug_csv", required=True, help="annotation worksheet with a confirm_positive column")
    ap.add_argument("--misses_csv", required=True, help="gene-present screen misses (pmid, title, abstract, g2p)")
    ap.add_argument("--ddg2p", required=True, help="G2P DD CSV")
    ap.add_argument("--out_csv", required=True, help="results CSV, one row per model")
    ap.add_argument("--aug_sizes", default=None,
                    help="comma-separated numbers of augmentation positives for a learning curve")
    ap.add_argument("--random_csv", default=None,
                    help="random PubMed sample (pmid, tiab); its positive rate is reported per model")
    ap.add_argument("--dump_scores", default=None,
                    help="directory for per-example positive-class probabilities (test, misses, random) per model")
    ap.add_argument("--dry_run", action="store_true", help="data preparation only, no training")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    from datasets import concatenate_datasets, load_from_disk
    ds_train = load_from_disk(f"{args.data_dir}/ds_bert_train")
    ds_test = load_from_disk(f"{args.data_dir}/ds_test")
    aug, _ = aug_ds(args.aug_csv, ds_train.features, lgmde_col=None)
    misses = load_misses(args.misses_csv, args.ddg2p)
    print(f"ds_bert_train {ds_train.num_rows} | aug {aug.num_rows} (pos {sum(aug['label'])}) | "
          f"ds_test {ds_test.num_rows} | misses {len(misses)} "
          f"(heldout-gene {int((misses.fold=='heldout').sum())}, train-gene {int((misses.fold=='train').sum())})")
    if args.dry_run:
        print("dry_run OK: datasets aligned, features cast, misses fold-tagged.")
        return

    rnd = pd.read_csv(args.random_csv, dtype=str).fillna("") if args.random_csv else None
    if rnd is not None:
        print(f"random PubMed sample: {len(rnd)}")

    import torch
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    rows = []
    if args.aug_sizes:
        # Learning curve: one model per number of augmentation positives.
        runs = [(n, ds_train if n == 0 else
                 concatenate_datasets([ds_train, aug_ds(args.aug_csv, ds_train.features, n_pos=n, lgmde_col=None)[0]]))
                for n in (int(x) for x in args.aug_sizes.split(","))]
        key = "n_aug_pos"
    else:
        runs = [("baseline", ds_train), ("augmented", concatenate_datasets([ds_train, aug]))]
        key = "variant"
    for label, tr_ds in runs:
        print(f"\n=== {key}={label}: train {tr_ds.num_rows} ===", flush=True)
        model, f1 = train_and_eval(tr_ds, ds_test, tokenizer, model_name=MODEL, hp=HP,
                                   out_dir=f"./_aug_{label}", logging_steps=100, max_length=MAX_LENGTH,
                                   bf16=torch.cuda.is_available(), accuracy=True)
        misses["pred"] = score_argmax(model, tokenizer, misses["tiab"], MAX_LENGTH)
        row = {key: label, **f1,
               "miss_recovery_pct": round(100 * misses["pred"].mean(), 1),
               "miss_recovery_heldout_pct": round(100 * misses.loc[misses.fold == "heldout", "pred"].mean(), 1),
               "miss_recovery_trainfold_pct": round(100 * misses.loc[misses.fold == "train", "pred"].mean(), 1)}
        if rnd is not None:
            rnd_pred = score_argmax(model, tokenizer, rnd["tiab"], MAX_LENGTH)
            row["random_pubmed_pos_rate_pct"] = round(100 * rnd_pred.mean(), 2)
        rows.append(row)
        print(rows[-1])
        if args.dump_scores:
            os.makedirs(args.dump_scores, exist_ok=True)
            pd.DataFrame({"label": list(ds_test["label"]),
                          "proba": score_proba(model, tokenizer, list(ds_test["tiab"]), MAX_LENGTH)}
                         ).to_csv(f"{args.dump_scores}/{label}_test.csv", index=False)
            pd.DataFrame({"fold": misses["fold"].values,
                          "proba": score_proba(model, tokenizer, misses["tiab"], MAX_LENGTH)}
                         ).to_csv(f"{args.dump_scores}/{label}_misses.csv", index=False)
            if rnd is not None:
                pd.DataFrame({"proba": score_proba(model, tokenizer, rnd["tiab"], MAX_LENGTH)}
                             ).to_csv(f"{args.dump_scores}/{label}_random.csv", index=False)
            print(f"  dumped scores -> {args.dump_scores}/{label}_{{test,misses,random}}.csv")
        del model
        gc.collect()
        torch.cuda.empty_cache()
    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print("\n=== results ===")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
