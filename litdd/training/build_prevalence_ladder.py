#!/usr/bin/env python3
"""Build training sets at several negative:positive ratios (the prevalence ladder).

The released screen was trained at 51.8% positive against a deployment prevalence of ~1-2%
and fires on 19.46% of random PubMed. This produces variants of the same training set with
increasing numbers of corpus-representative negatives, so the recall/corpus-rate trade can be
measured rather than guessed.

Each arm keeps the positives and the original in-domain negatives untouched and only ADDS
corpus negatives, so the arms differ in exactly one thing.
"""
import argparse, csv, os, sys
csv.field_size_limit(10**9)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train_ds_dir", default="revision/external_recall/ds_hirecall_train")
    ap.add_argument("--negatives_csv", default="revision/external_recall/corpus_negatives.csv")
    ap.add_argument("--add", nargs="+", type=int, default=[0, 20000, 60000, 150000])
    ap.add_argument("--out_root", default="revision/external_recall/ladder")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    from datasets import load_from_disk, Dataset, concatenate_datasets
    import random

    base = load_from_disk(a.train_ds_dir)
    pos = sum(1 for x in base["label"] if x == 1)
    print(f"base: {len(base):,} rows, {pos:,} positive ({100*pos/len(base):.1f}%)")

    negs = list(csv.DictReader(open(a.negatives_csv, newline="", encoding="utf-8")))
    random.seed(a.seed); random.shuffle(negs)
    print(f"corpus negatives available: {len(negs):,}")

    cols = base.column_names
    for n in a.add:
        if n > len(negs):
            print(f"[warn] arm +{n} exceeds pool ({len(negs):,}); skipping"); continue
        if n == 0:
            ds = base
        else:
            chunk = negs[:n]
            # Match the base schema exactly; g2p_lgmde is empty for corpus negatives because
            # they are screen-level (TIAB-only) negatives with no candidate entry attached.
            extra = Dataset.from_dict({
                "tiab": [r["tiab"] for r in chunk],
                "g2p_lgmde": ["" for _ in chunk],
                "label": [0 for _ in chunk],
            })
            extra = extra.select_columns([c for c in cols if c in extra.column_names])
            # The base set types `label` as ClassLabel, not int64, so concatenation refuses
            # to align features unless the new rows are cast to the same feature spec.
            extra = extra.cast(base.features)
            ds = concatenate_datasets([base, extra]).shuffle(seed=a.seed)
        p = sum(1 for x in ds["label"] if x == 1)
        out = os.path.join(a.out_root, f"add{n}")
        ds.save_to_disk(out)
        print(f"  +{n:<7,} -> {len(ds):>8,} rows, {100*p/len(ds):>5.1f}% positive  {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
