#!/usr/bin/env python3
"""Build train / test splits for the LitDD-BERT classifier.

The split is performed at the PMID-group level — every grouping key (default
``pmid``) appears in exactly one of {train, test}, with stratification on
whether that group has any positive label. Default ratio is 80 / 20.

Hyperparameters are selected via 5-fold ``StratifiedGroupKFold`` cross-
validation **inside** the train portion (see ``cross_validation/`` scripts).
The test set is touched exactly once, after refitting on the full train with
the selected hyperparameters.

Inputs:
  --annotated_csv   CSV with columns: pmid, tiab (optional), g2p_lgmde, label.
                    Defaults to ``annotated_pmid.csv`` shipped with the repo.

Outputs (HuggingFace ``save_to_disk`` directories):
  ds_bert_train, ds_cross_train  — train portion (BERT and cross-encoder
                                    share the same training examples)
  ds_test                         — held-out test, only touched after the
                                    final refit

Use ``--dry_run`` to print sizes / ratios without writing any files.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
from datasets import ClassLabel, Dataset, Features, Value
from sklearn.model_selection import train_test_split

REQUIRED_COLS = {"label"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--annotated_csv",
        default="train_test/annotated_pmid.csv",
        help="Input CSV with at least 'g2p_lgmde' and 'label' columns; ideally also 'tiab' or 'pmid'.",
    )
    p.add_argument("--out_dir", default="train_test", help="Directory to write ds_*/ subdirs.")
    p.add_argument("--test_size", type=float, default=0.20, help="Test fraction.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--group_col",
        default="tiab",
        help="Column to group on to prevent leakage. Falls back to 'pmid' when 'tiab' is absent.",
    )
    p.add_argument("--dry_run", action="store_true", help="Print sizes; do not write to disk.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    df = pd.read_csv(args.annotated_csv)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        print(f"[ERROR] {args.annotated_csv} missing columns: {missing}", file=sys.stderr)
        return 1

    group_col = args.group_col if args.group_col in df.columns else "pmid"
    if group_col not in df.columns:
        print(f"[ERROR] No group column ('{args.group_col}' or 'pmid') in input.", file=sys.stderr)
        return 1
    if "g2p_lgmde" not in df.columns:
        print(f"[ERROR] {args.annotated_csv} missing 'g2p_lgmde' column.", file=sys.stderr)
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
    assert train_keys.isdisjoint(test_keys), f"{group_col} leakage: train ↔ test"

    df_train = df[df[group_col].isin(train_keys)].copy()
    df_test = df[df[group_col].isin(test_keys)].copy()

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

    feature_kwargs = {"label": ClassLabel(num_classes=2)}
    if "tiab" in df_train.columns:
        feature_kwargs["tiab"] = Value("string")
    feature_kwargs["g2p_lgmde"] = Value("string")
    features = Features(feature_kwargs)

    ds_train = Dataset.from_pandas(df_train, preserve_index=False).cast(features)
    ds_test = Dataset.from_pandas(df_test, preserve_index=False).cast(features)

    os.makedirs(args.out_dir, exist_ok=True)
    train_out = os.path.join(args.out_dir, "ds_bert_train")
    cross_out = os.path.join(args.out_dir, "ds_cross_train")
    test_out = os.path.join(args.out_dir, "ds_test")

    ds_train.save_to_disk(train_out)
    ds_train.save_to_disk(cross_out)  # cross-encoder uses the same train set
    ds_test.save_to_disk(test_out)

    print(f"[Info] saved → {train_out}, {cross_out}, {test_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
