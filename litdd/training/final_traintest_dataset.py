#!/usr/bin/env python3
"""Split the annotated set into training and test portions at the group level.

Reads a CSV with at least ``g2p_lgmde`` and ``label`` columns (``tiab`` and ``pmid`` optional).
Every value of the grouping column falls in exactly one of the two portions, and the split is
stratified on whether a group holds any positive label. ``--group_col`` chooses the grouping
axis:

  tiab    (default) no abstract appears in both portions
  pmid              no PMID appears in both portions
  gene              no gene appears in both portions
  g2p_id            no G2P disease entry appears in both portions

``gene`` and ``g2p_id`` are parsed from ``g2p_lgmde`` and used for grouping only; they are not
written to the output. When the requested column is absent the split falls back to ``pmid``.

Writes two HuggingFace ``save_to_disk`` directories under ``--out_dir``: ``ds_bert_train`` and
``ds_test``, each with ``tiab`` (when present), ``g2p_lgmde`` and a two-class ``label``.
Hyperparameters are then selected by cross-validation inside ``ds_bert_train``
(``cv_hp_search_bert.py``) and ``ds_test`` is evaluated once after the final refit.
``--dry_run`` prints the group and row counts without writing.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

# datasets and sklearn are imported inside main() so derive_group_columns can be
# imported and tested without them.

REQUIRED_COLS = {"label"}
# g2p_lgmde fields: g2p_id - gene symbol - gene_mim - hgnc - prev_symbols - disease - ...
LGMDE_GENE_FIELD = 1


def derive_group_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with ``g2p_id`` and ``gene`` columns parsed from ``g2p_lgmde``."""
    parts = df["g2p_lgmde"].astype(str).str.split(" - ")
    df = df.copy()
    df["g2p_id"] = parts.map(lambda p: p[0].strip() if p else "")
    df["gene"] = parts.map(lambda p: p[LGMDE_GENE_FIELD].strip() if len(p) > LGMDE_GENE_FIELD else "")
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--annotated_csv",
        default="data/annotated_pmid.csv",
        help="Input CSV with at least 'g2p_lgmde' and 'label' columns; ideally also 'tiab' or 'pmid'.",
    )
    p.add_argument("--out_dir", default="data", help="Directory to write ds_*/ subdirs.")
    p.add_argument("--test_size", type=float, default=0.20, help="Test fraction.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--group_col",
        default="tiab",
        choices=["tiab", "pmid", "gene", "g2p_id"],
        help="Grouping axis: tiab, pmid, gene (gene-held-out) or g2p_id (disease-held-out). "
             "Falls back to 'pmid' when the column is absent.",
    )
    p.add_argument("--dry_run", action="store_true", help="Print sizes; do not write to disk.")
    return p.parse_args()


def main() -> int:
    from sklearn.model_selection import train_test_split

    args = parse_args()

    df = pd.read_csv(args.annotated_csv)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        print(f"[ERROR] {args.annotated_csv} missing columns: {missing}", file=sys.stderr)
        return 1

    if "g2p_lgmde" not in df.columns:
        print(f"[ERROR] {args.annotated_csv} missing 'g2p_lgmde' column.", file=sys.stderr)
        return 1

    # Parse gene and g2p_id, then resolve the grouping column.
    df = derive_group_columns(df)
    group_col = args.group_col if args.group_col in df.columns else "pmid"
    if group_col not in df.columns:
        print(f"[ERROR] No group column ('{args.group_col}' or 'pmid') in input.", file=sys.stderr)
        return 1

    grp = df.groupby(group_col, as_index=False)["label"].max()
    grp.rename(columns={"label": "has_pos"}, inplace=True)

    train_grp, test_grp = train_test_split(
        grp,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=grp["has_pos"],
    )
    train_keys = set(train_grp[group_col])
    test_keys = set(test_grp[group_col])
    assert train_keys.isdisjoint(test_keys), f"{group_col} shared between train and test"

    df_train = df[df[group_col].isin(train_keys)].copy()
    df_test = df[df[group_col].isin(test_keys)].copy()

    print(f"[Info] group_col='{group_col}' — {len(train_keys)} train / {len(test_keys)} test disjoint groups.")
    if group_col in ("gene", "g2p_id") and "tiab" in df.columns:
        # An abstract can pair with a held-out candidate and a retained one, so it may appear on both sides.
        shared = set(df_train["tiab"]) & set(df_test["tiab"])
        print(f"[Info] {group_col}-held-out: {len(shared)} abstract(s) appear on both sides.")

    keep = ["tiab", "g2p_lgmde", "label"] if "tiab" in df.columns else ["g2p_lgmde", "label"]
    df_train = df_train[keep]
    df_test = df_test[keep]

    n_total = len(grp)
    print(f"[Info] groups: total={n_total} "
          f"train={len(train_grp)} ({len(train_grp)/n_total:.1%}) "
          f"test={len(test_grp)} ({len(test_grp)/n_total:.1%})")
    print(f"[Info] rows:   train={len(df_train)} test={len(df_test)}")
    print(f"[Info] has_pos rate — "
          f"train={df_train['label'].mean():.3f} test={df_test['label'].mean():.3f}")

    if args.dry_run:
        print("[Info] --dry_run set; not writing datasets to disk.")
        return 0

    from datasets import ClassLabel, Dataset, Features, Value

    feature_kwargs = {"label": ClassLabel(num_classes=2)}
    if "tiab" in df_train.columns:
        feature_kwargs["tiab"] = Value("string")
    feature_kwargs["g2p_lgmde"] = Value("string")
    features = Features(feature_kwargs)

    ds_train = Dataset.from_pandas(df_train, preserve_index=False).cast(features)
    ds_test = Dataset.from_pandas(df_test, preserve_index=False).cast(features)

    os.makedirs(args.out_dir, exist_ok=True)
    train_out = os.path.join(args.out_dir, "ds_bert_train")
    test_out = os.path.join(args.out_dir, "ds_test")

    ds_train.save_to_disk(train_out)
    ds_test.save_to_disk(test_out)

    print(f"[Info] saved → {train_out}, {test_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
