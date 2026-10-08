#!/usr/bin/env python3
"""Build the screen's training set before corpus negatives: annotated training split, confirmed
worksheet rows and the curated external positives.

Reads ``ds_bert_train`` from ``--data_dir`` (from ``final_traintest_dataset.py``), an annotation
worksheet (``--aug_csv``, rows with a ``confirm_positive`` token) and the curated external
papers (``--external_csv``: ``pmid, tiab, g2p_id, gene, source``; DDG2P publications, HPO
annotations and ClinGen case-level evidence).

External papers are deduplicated by PMID (first row kept) and the gene of the kept row is
bucketed with ``gene_fold(gene, 10)``, as in the run that produced the released screen.
Papers whose genes all fall in bucket 0 are held out for the recall evaluation; papers mixing
bucket 0 with other buckets are dropped; every other paper is added as a positive. The result
is written with ``save_to_disk`` to ``--out_dir`` and is the ``--train_ds_dir`` of
``build_prevalence_ladder.py``, which adds the corpus negatives (the released screen uses the
``add20000`` arm).

Example::

    python -m litdd.training.build_screen_training_set --data_dir data \\
        --aug_csv revision/external_recall/250_augmentation_candidates_to_annotate.csv \\
        --external_csv revision/external_recall/external_positives.csv \\
        --out_dir data/ds_screen_train
"""
from __future__ import annotations

import argparse

import pandas as pd

from litdd.training.screen_common import as_positive_ds, aug_ds, gene_fold

HELDOUT_MODULUS = 10


def split_external(ext: pd.DataFrame, modulus: int = HELDOUT_MODULUS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(trainable, heldout)`` external papers, one row per PMID.

    Papers are deduplicated by PMID first, so each is bucketed on the gene of its first row:
    held out when that gene is in bucket 0, trainable otherwise.
    """
    ext = ext.drop_duplicates("pmid").copy()
    buckets = ext.groupby("pmid")["gene"].apply(lambda gs: {gene_fold(g, modulus) for g in gs})
    b = ext["pmid"].map(buckets)
    heldout = ext[b.map(lambda s: s == {0})]
    trainable = ext[b.map(lambda s: 0 not in s)]
    return trainable.reset_index(drop=True), heldout.reset_index(drop=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True, help="directory with ds_bert_train")
    ap.add_argument("--aug_csv", required=True, help="annotation worksheet with a confirm_positive column")
    ap.add_argument("--external_csv", required=True, help="curated external papers (pmid, tiab, g2p_id, gene, source)")
    ap.add_argument("--out_dir", required=True, help="output training set (save_to_disk directory)")
    ap.add_argument("--dry_run", action="store_true", help="print the composition and exit")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from datasets import concatenate_datasets, load_from_disk

    ds_train = load_from_disk(f"{args.data_dir}/ds_bert_train")
    ext = pd.read_csv(args.external_csv, dtype=str)
    trainable, heldout = split_external(ext)
    aug, _ = aug_ds(args.aug_csv, ds_train.features)
    print(f"annotated training split {ds_train.num_rows} | confirmed worksheet rows {aug.num_rows} | "
          f"external positives {len(trainable)} ({trainable['source'].value_counts().to_dict()}) | "
          f"external held out for recall {len(heldout)}")
    if args.dry_run:
        return 0
    out = concatenate_datasets([ds_train, aug, as_positive_ds(trainable, ds_train.features)])
    out.save_to_disk(args.out_dir)
    print(f"wrote {args.out_dir}: {out.num_rows} rows, {sum(out['label'])} positive")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
