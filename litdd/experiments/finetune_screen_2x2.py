#!/usr/bin/env python3
"""Train the screen in a 2x2 design: augmentation off/on by gene-conditioning off/on.

Reads ``ds_bert_train`` and ``ds_test`` from ``--data_dir``, an annotation worksheet
(``--aug_csv``), the gene-present screen misses (``--misses_csv``: ``pmid, title, abstract,
g2p``), the G2P DD CSV (``--ddg2p``, for gene symbols and previous symbols) and NCBI
``gene_info.gz`` (``--gene_info``, for approved full names).

Augmentation appends the confirmed worksheet rows to the training set. Gene-conditioning
tokenises each example as the pair (abstract, "symbol ; previous symbols ; full name") of its
candidate gene; without it the gene is ignored. Every cell is evaluated on ``ds_test``
(precision, recall, F1) and on the misses (share predicted positive, overall and split by
whether the miss's gene falls in held-out fold 0 of 5). Writes one row per cell to
``--out_csv``.
"""
from __future__ import annotations

import argparse
import gc

import pandas as pd

from litdd.training.screen_common import (
    confirmed_worksheet,
    fold_name,
    gene_fullnames,
    score_argmax,
    train_and_eval,
)

MODEL = "thomas-sounack/BioClinical-ModernBERT-large"
HP = dict(epochs=5, bs=32, lr=1.736e-5, wd=0.3)
MAX_LENGTH = 512


def cond_of(gene: str, prev: str, names: dict[str, str]) -> str:
    """Conditioning string: the symbol, its previous symbols and its full name, de-duplicated."""
    forms = [gene] + [x.strip() for x in prev.replace(";", ",").split(",") if x.strip()]
    if names.get(gene):
        forms.append(names[gene])
    return " ; ".join(dict.fromkeys(f for f in forms if f))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True, help="directory with ds_bert_train and ds_test")
    ap.add_argument("--aug_csv", required=True, help="annotation worksheet with a confirm_positive column")
    ap.add_argument("--misses_csv", required=True, help="gene-present screen misses (pmid, title, abstract, g2p)")
    ap.add_argument("--ddg2p", required=True, help="G2P DD CSV")
    ap.add_argument("--gene_info", required=True, help="NCBI gene_info.gz")
    ap.add_argument("--out_csv", required=True, help="results CSV, one row per cell")
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    from datasets import Dataset, load_from_disk
    dd = pd.read_csv(args.ddg2p, dtype=str).fillna("")
    dd.columns = [c.strip() for c in dd.columns]
    g2gene = dict(zip(dd["g2p id"], dd["gene symbol"].str.strip()))
    g2prev = dict(zip(dd["g2p id"], dd["previous gene symbols"]))
    names = gene_fullnames(args.gene_info)

    def lgmde_cond(s: str) -> str:
        # g2p_lgmde fields: gene symbol at index 1, previous symbols at index 4.
        p = s.split(" - ")
        return cond_of(p[1].strip() if len(p) > 1 else "", p[4] if len(p) > 4 else "", names)

    def build(ds) -> pd.DataFrame:
        df = ds.to_pandas()
        df["cond"] = df["g2p_lgmde"].map(lgmde_cond)
        df["label"] = df["label"].astype(int)
        return df[["tiab", "cond", "label"]]
    train_base = build(load_from_disk(f"{args.data_dir}/ds_bert_train"))
    test_df = build(load_from_disk(f"{args.data_dir}/ds_test"))

    a = confirmed_worksheet(pd.read_csv(args.aug_csv, dtype=str).fillna(""))
    a["cond"] = [cond_of(g, g2prev.get(gid, ""), names) for g, gid in zip(a["gene"], a["g2p_id"])]
    aug = a[["tiab", "cond", "label"]]

    m = pd.read_csv(args.misses_csv, dtype=str).fillna("").drop_duplicates("pmid")
    m["tiab"] = (m["title"] + " " + m["abstract"]).str.strip()
    m["gene"] = m["g2p"].map(lambda g: g2gene.get(g, ""))
    m["cond"] = [cond_of(g2gene.get(g, ""), g2prev.get(g, ""), names) for g in m["g2p"]]
    m["foldg"] = m["gene"].map(lambda g: fold_name(g, 5))

    print(f"train_base {len(train_base)} | aug {len(aug)} (pos {int(aug.label.sum())}) | test {len(test_df)} | "
          f"misses {len(m)} (heldout {int((m.foldg=='heldout').sum())})")
    if args.dry_run:
        print("dry_run OK")
        return

    import torch
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    def as_ds(df: pd.DataFrame):
        return Dataset.from_pandas(df[["tiab", "cond", "label"]].reset_index(drop=True))

    rows = []
    for gcond in (False, True):
        for use_aug in (False, True):
            tr_df = pd.concat([train_base, aug], ignore_index=True) if use_aug else train_base
            name = f"{'genecond' if gcond else 'plain'}{'+aug' if use_aug else ''}"
            print(f"\n=== {name}: train {len(tr_df)} | gene_cond={gcond} aug={use_aug} ===", flush=True)
            model, f1 = train_and_eval(as_ds(tr_df), as_ds(test_df), tokenizer, model_name=MODEL, hp=HP,
                                       out_dir=f"./_2x2_{name}", logging_steps=200, max_length=MAX_LENGTH,
                                       bf16=torch.cuda.is_available(), pair_col="cond" if gcond else None)
            m["pred"] = score_argmax(model, tokenizer, list(m["tiab"]), MAX_LENGTH,
                                     pair_texts=list(m["cond"]) if gcond else None)
            rows.append({"cell": name, "gene_conditioned": gcond, "augmented": use_aug, **f1,
                         "miss_recovery_pct": round(100 * m["pred"].mean(), 1),
                         "miss_recovery_heldout_pct": round(100 * m.loc[m.foldg == "heldout", "pred"].mean(), 1),
                         "miss_recovery_trainfold_pct": round(100 * m.loc[m.foldg == "train", "pred"].mean(), 1)})
            print(rows[-1])
            del model
            gc.collect()
            torch.cuda.empty_cache()
    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print("\n=== 2x2 results ===")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
