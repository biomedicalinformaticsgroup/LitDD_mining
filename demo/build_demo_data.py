#!/usr/bin/env python3
"""Build the small demo dataset from data/annotated_pmid.csv.

Stratified-samples 100 PMIDs (group-level, preserving the positive-class
rate of the full annotated dataset) and writes ``demo/data/annotated_pmid_demo.csv``.

Also stages a tiny G2P CSV from clean_pipeline that covers the G2P IDs
referenced by the sampled rows, so mine_hard_negatives.py / the
cross-encoder CV search can build the candidate corpus end-to-end on CPU.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SRC_ANNOTATED = ROOT / "train_test" / "annotated_pmid.csv"
# Point at the reference clean_pipeline via LITDD_REF_DIR (no absolute path shipped).
SRC_G2P = Path(os.environ.get("LITDD_REF_DIR", "reference_data/clean_pipeline")) / "train_test" / "G2P_DD_2025-02-15.csv"
OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(parents=True, exist_ok=True)

N_DEMO = 100
SEED = 42


def main() -> None:
    df = pd.read_csv(SRC_ANNOTATED)
    print(f"[1/3] full annotated set: {len(df)} rows; positive rate {df['label'].mean():.3f}")

    # Group-stratified sample of 100 PMIDs
    grp = df.groupby("pmid", as_index=False)["label"].max()
    grp.rename(columns={"label": "has_pos"}, inplace=True)
    sampled, _ = train_test_split(
        grp, train_size=N_DEMO, random_state=SEED, stratify=grp["has_pos"]
    )
    sampled_pmids = set(sampled["pmid"])
    df_demo = df[df["pmid"].isin(sampled_pmids)].copy()
    out_csv = OUT / "annotated_pmid_demo.csv"
    df_demo.to_csv(out_csv, index=False)
    print(f"[2/3] demo annotated set: {len(df_demo)} rows over {df_demo['pmid'].nunique()} PMIDs; "
          f"positive rate {df_demo['label'].mean():.3f} → {out_csv}")

    # Stage a tiny G2P CSV containing every G2P_ID referenced in the demo rows
    referenced = set()
    for v in df_demo["g2p_lgmde"]:
        referenced.add(str(v).split(" - ", 1)[0].strip())
    g2p_full = pd.read_csv(SRC_G2P, dtype=str, keep_default_na=False)
    g2p_demo = g2p_full[g2p_full["g2p id"].isin(referenced)].copy()
    # Add 50 unrelated G2P rows so the cross-encoder has a meaningful candidate corpus
    extras = g2p_full[~g2p_full["g2p id"].isin(referenced)].head(50)
    g2p_out = pd.concat([g2p_demo, extras], ignore_index=True)
    out_g2p = OUT / "g2p_demo.csv"
    g2p_out.to_csv(out_g2p, index=False)
    print(f"[3/3] demo G2P corpus: {len(g2p_out)} rows ({len(g2p_demo)} referenced + 50 extras) → {out_g2p}")


if __name__ == "__main__":
    main()
