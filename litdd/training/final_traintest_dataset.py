#!/usr/bin/env python3
"""Build train / test splits for the LitDD-BERT classifier.

The split is performed at the group level — every grouping key appears in exactly
one of {train, test}, with stratification on whether that group has any positive
label. Default ratio is 80 / 20.

``--group_col`` selects the leakage-control axis and supports stricter held-out
validations (Reviewer 2 E1/E2):
  tiab    (default) — no abstract shared across train/test
  pmid              — no PMID shared
  gene              — GENE-held-out: no gene appears in both halves
  g2p_id            — DISEASE-held-out: no G2P disease entry appears in both halves
``gene`` and ``g2p_id`` are derived from ``g2p_lgmde`` and used only for grouping
(never as model features). Gene-/disease-held-out are the stricter generalisation
tests the reviewer asks for; a TIAB-level split can overestimate when the same
gene/disease context appears on both sides.

Hyperparameters are selected via 5-fold ``StratifiedGroupKFold`` cross-
validation **inside** the train portion (see ``litdd/training/`` scripts).
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

# datasets + sklearn are imported lazily inside main() so the pure helpers
# (e.g. derive_group_columns) can be imported/tested without the heavy deps.

REQUIRED_COLS = {"label"}
# g2p_lgmde format: g2p_id - gene symbol - gene_mim - hgnc - prev_symbols - disease - ...
LGMDE_GENE_FIELD = 1


def derive_group_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add `gene` and `g2p_id` columns parsed from g2p_lgmde (for held-out grouping)."""
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
        help="Leakage-control axis: tiab/pmid, or gene (gene-held-out) / g2p_id "
             "(disease-held-out). Falls back to 'pmid' when the column is absent.",
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

    # derive gene / g2p_id for the stricter held-out splits, then resolve the group column
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
    assert train_keys.isdisjoint(test_keys), f"{group_col} leakage: train ↔ test"

    df_train = df[df[group_col].isin(train_keys)].copy()
    df_test = df[df[group_col].isin(test_keys)].copy()

    print(f"[Info] group_col='{group_col}' — {len(train_keys)} train / {len(test_keys)} test "
          f"disjoint groups (held-out).")
    if group_col in ("gene", "g2p_id") and "tiab" in df.columns:
        shared = set(df_train["tiab"]) & set(df_test["tiab"])
        print(f"[Info] {group_col}-held-out: {len(shared)} abstract(s) appear on both sides "
              "(expected — an abstract can pair with held-out and retained candidates).")

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
    cross_out = os.path.join(args.out_dir, "ds_cross_train")
    test_out = os.path.join(args.out_dir, "ds_test")

    ds_train.save_to_disk(train_out)
    ds_train.save_to_disk(cross_out)  # cross-encoder uses the same train set
    ds_test.save_to_disk(test_out)

    print(f"[Info] saved → {train_out}, {cross_out}, {test_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
