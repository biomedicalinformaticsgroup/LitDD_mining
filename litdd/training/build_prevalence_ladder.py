#!/usr/bin/env python3
"""Build training sets that add increasing numbers of corpus negatives to a base set.

Reads a training dataset (``--train_ds_dir``, a HuggingFace ``save_to_disk`` directory with
``tiab``, ``g2p_lgmde`` and a two-class ``label``) and the negatives CSV written by
``build_corpus_negatives.py`` (``--negatives_csv``). For each count in ``--add`` it shuffles the
negatives with ``--seed``, takes the first ``n`` as rows with empty ``g2p_lgmde`` and label 0,
casts them to the base features, concatenates and shuffles, and saves the result to
``<out_root>/add<n>``. The positives and the base negatives are identical across arms, so the
arms differ only in the number of corpus negatives. Prints the row count and positive share
of each arm.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

csv.field_size_limit(10**9)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train_ds_dir", required=True, help="base training dataset (save_to_disk directory)")
    ap.add_argument("--negatives_csv", required=True, help="corpus negatives from build_corpus_negatives.py")
    ap.add_argument("--add", nargs="+", type=int, default=[0, 20000, 60000, 150000],
                    help="numbers of corpus negatives to add, one arm each")
    ap.add_argument("--out_root", required=True, help="directory receiving one add<n>/ dataset per arm")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    import random

    from datasets import Dataset, concatenate_datasets, load_from_disk

    base = load_from_disk(a.train_ds_dir)
    pos = sum(1 for x in base["label"] if x == 1)
    print(f"base: {len(base):,} rows, {pos:,} positive ({100*pos/len(base):.1f}%)")

    with open(a.negatives_csv, newline="", encoding="utf-8") as f:
        negs = list(csv.DictReader(f))
    random.seed(a.seed)
    random.shuffle(negs)
    print(f"corpus negatives available: {len(negs):,}")

    cols = base.column_names
    for n in a.add:
        if n > len(negs):
            print(f"[warn] arm +{n} exceeds pool ({len(negs):,}); skipping")
            continue
        if n == 0:
            ds = base
        else:
            chunk = negs[:n]
            # Corpus negatives carry no candidate entry, so g2p_lgmde is empty.
            extra = Dataset.from_dict({
                "tiab": [r["tiab"] for r in chunk],
                "g2p_lgmde": ["" for _ in chunk],
                "label": [0 for _ in chunk],
            })
            extra = extra.select_columns([c for c in cols if c in extra.column_names])
            # Cast label to the base ClassLabel feature so the two datasets concatenate.
            extra = extra.cast(base.features)
            ds = concatenate_datasets([base, extra]).shuffle(seed=a.seed)
        p = sum(1 for x in ds["label"] if x == 1)
        out = os.path.join(a.out_root, f"add{n}")
        ds.save_to_disk(out)
        print(f"  +{n:<7,} -> {len(ds):>8,} rows, {100*p/len(ds):>5.1f}% positive  {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
