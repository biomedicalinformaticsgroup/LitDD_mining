#!/usr/bin/env python3
"""Sample the training/annotation set for a second-annotator inter-annotator-agreement
(IAA) exercise on the ORIGINAL labels (Reviewer 3 R3.10 / Reviewer 2 R2-A1).

The ground-truth labels were produced by a single clinical geneticist. To quantify
annotation reliability, a second annotator re-labels a blinded sample of (abstract,
candidate G2P disease) pairs drawn from ``annotated_pmid.csv`` (pmid, g2p_lgmde, label);
Cohen's kappa vs the original labels is then computed by score_audit.py.

This is distinct from the deployed-corpus precision audit (sample_audit.py): there the
unit is a pipeline *output* mapping judged correct/incorrect; here the unit is a *training
label* (does this abstract support the candidate disease? 1/0), re-applied independently.

Abstracts are not in annotated_pmid.csv, so a --tiab_source parquet (pmid + tiab, or
title/abstract) is joined in. Output goes to the gitignored revision/ area.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LGMDE_DISEASE_FIELD = 5


def load_tiab(tiab_source: str) -> pd.DataFrame:
    """Return a (pmid, tiab) frame from a CSV/parquet that has tiab or title+abstract."""
    df = pd.read_csv(tiab_source) if str(tiab_source).endswith(".csv") else pd.read_parquet(tiab_source)
    df = df.rename(columns={c: c.lower() for c in df.columns})
    if "tiab" not in df.columns:
        title = df.get("title", "").fillna("") if "title" in df.columns else ""
        abstract = df.get("abstract", "").fillna("") if "abstract" in df.columns else ""
        df["tiab"] = (title + " " + abstract).str.strip()
    df["pmid"] = df["pmid"].astype(int)
    return df[["pmid", "tiab"]].drop_duplicates("pmid")


def stratified_by_label(df: pd.DataFrame, n: int, balanced: bool, rng) -> pd.DataFrame:
    if balanced:
        per = n // 2
        parts = []
        for lab in (0, 1):
            cell = df[df["label"] == lab]
            parts.append(cell.sample(n=min(per, len(cell)), random_state=rng.integers(1 << 31)))
        out = pd.concat(parts)
        if len(out) < n:  # top up if a class is small
            extra = df.drop(index=out.index).sample(n=min(n - len(out), len(df) - len(out)),
                                                     random_state=rng.integers(1 << 31))
            out = pd.concat([out, extra])
    else:
        out = df.sample(n=min(n, len(df)), random_state=rng.integers(1 << 31))
    return out.sample(frac=1, random_state=rng.integers(1 << 31)).reset_index(drop=True)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotated_csv", default="train_test/annotated_pmid.csv",
                    help="pmid, g2p_lgmde, label (and optionally tiab)")
    ap.add_argument("--tiab_source", default=None,
                    help="CSV/parquet with pmid + tiab (or title/abstract); needed only if "
                         "--annotated_csv has no tiab column")
    ap.add_argument("--out_dir", default="revision/precision_audit")
    ap.add_argument("--n", type=int, default=100, help="IAA sample size")
    ap.add_argument("--balanced", action="store_true", default=True,
                    help="Sample equal positive/negative labels (default)")
    ap.add_argument("--proportional", dest="balanced", action="store_false",
                    help="Sample proportional to the label distribution instead")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    ann = pd.read_csv(args.annotated_csv)
    ann["pmid"] = ann["pmid"].astype(int)
    if "tiab" in ann.columns:
        merged = ann
    else:
        if not args.tiab_source:
            raise SystemExit("--annotated_csv has no 'tiab' column; pass --tiab_source.")
        merged = ann.merge(load_tiab(args.tiab_source), on="pmid", how="left")
    missing = merged["tiab"].isna().sum()
    if missing:
        print(f"WARNING: {missing}/{len(merged)} annotated rows have no tiab; dropping them.")
        merged = merged[merged["tiab"].notna()]

    sample = stratified_by_label(merged, args.n, args.balanced, rng)
    sample.insert(0, "iaa_id", [f"T{i:03d}" for i in range(len(sample))])
    # readable candidate disease/gene from the LGMDE thread
    parts = sample["g2p_lgmde"].astype(str).str.split(" - ")
    sample["candidate_gene"] = parts.map(lambda p: p[1] if len(p) > 1 else "")
    sample["candidate_disease"] = parts.map(lambda p: p[LGMDE_DISEASE_FIELD] if len(p) > LGMDE_DISEASE_FIELD else "")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    worksheet = sample[["iaa_id", "pmid", "tiab", "candidate_gene", "candidate_disease", "g2p_lgmde"]].copy()
    worksheet["relevant"] = ""   # annotator B fills: 1 (abstract supports this G2P disease) | 0 | uncertain
    worksheet["notes"] = ""
    worksheet.to_csv(out / "trainlabel_iaa_worksheet.csv", index=False)

    sample[["iaa_id", "pmid", "g2p_lgmde", "label"]].rename(columns={"label": "original_label"}) \
        .to_csv(out / "trainlabel_iaa_key.csv", index=False)

    print(f"Wrote {len(worksheet)} IAA units to {out}/trainlabel_iaa_worksheet.csv")
    print("Original label distribution in sample:", dict(sample["label"].value_counts()))
    print("relevant = 1 (abstract supports the candidate G2P disease) | 0 | uncertain")


if __name__ == "__main__":
    main()
