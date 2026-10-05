#!/usr/bin/env python3
"""Build the demo dataset: a stratified sample of annotated PMIDs and the G2P entries they reference.

Reads the annotated CSV (``--annotated_csv``, default ``data/annotated_pmid.csv``) and the G2P DD
CSV (``--g2p_csv``). Samples ``--n`` PMIDs at the group level with the positive-class rate of the
full set preserved, and writes their rows to ``<out_dir>/annotated_pmid_demo.csv``. Writes
``<out_dir>/g2p_demo.csv`` with every G2P entry referenced by the sampled rows plus the first 50
unreferenced entries, so the demo's candidate set is not limited to the referenced entries.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

N_EXTRA_G2P = 50


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotated_csv", default="data/annotated_pmid.csv",
                    help="annotated CSV with pmid, g2p_lgmde and label columns")
    ap.add_argument("--g2p_csv", required=True, help="G2P DD CSV")
    ap.add_argument("--out_dir", default=str(Path(__file__).resolve().parent / "data"),
                    help="directory receiving annotated_pmid_demo.csv and g2p_demo.csv")
    ap.add_argument("--n", type=int, default=100, help="PMIDs to sample")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> None:
    from sklearn.model_selection import train_test_split

    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.annotated_csv)
    print(f"[1/3] full annotated set: {len(df)} rows; positive rate {df['label'].mean():.3f}")

    # Group-level stratified sample of PMIDs.
    grp = df.groupby("pmid", as_index=False)["label"].max()
    grp.rename(columns={"label": "has_pos"}, inplace=True)
    sampled, _ = train_test_split(grp, train_size=args.n, random_state=args.seed, stratify=grp["has_pos"])
    sampled_pmids = set(sampled["pmid"])
    df_demo = df[df["pmid"].isin(sampled_pmids)].copy()
    out_csv = out / "annotated_pmid_demo.csv"
    df_demo.to_csv(out_csv, index=False)
    print(f"[2/3] demo annotated set: {len(df_demo)} rows over {df_demo['pmid'].nunique()} PMIDs; "
          f"positive rate {df_demo['label'].mean():.3f} → {out_csv}")

    # G2P entries referenced by the demo rows, plus unreferenced entries.
    referenced = {str(v).split(" - ", 1)[0].strip() for v in df_demo["g2p_lgmde"]}
    g2p_full = pd.read_csv(args.g2p_csv, dtype=str, keep_default_na=False)
    g2p_demo = g2p_full[g2p_full["g2p id"].isin(referenced)].copy()
    extras = g2p_full[~g2p_full["g2p id"].isin(referenced)].head(N_EXTRA_G2P)
    g2p_out = pd.concat([g2p_demo, extras], ignore_index=True)
    out_g2p = out / "g2p_demo.csv"
    g2p_out.to_csv(out_g2p, index=False)
    print(f"[3/3] demo G2P entries: {len(g2p_out)} rows ({len(g2p_demo)} referenced + {N_EXTRA_G2P} extras) "
          f"→ {out_g2p}")


if __name__ == "__main__":
    main()
