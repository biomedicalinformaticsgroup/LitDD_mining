#!/usr/bin/env python3
"""Merge augmentation annotations into the BERT-screen training set, with the per-PMID
collapse rule (Reviewer 3 R3.4 / R1.3 augmentation).

The BERT screen is gene-AGNOSTIC ("is this a GDD gene-disease paper?", not "which disease"),
so a PMID must not be both positive and negative. Collapse rule, applied per PMID:
  - if ANY annotation for the PMID is label 1  -> keep the positive row(s), drop the 0 rows;
  - if ALL annotations for the PMID are 0       -> keep them all.
(The existing annotated set already satisfies this; this enforces it when new annotations are
added, e.g. if a paper is annotated against several candidate diseases.)

Labelling rule for the augmentation (gene level, for the screen):
  1 = the abstract shows THIS GENE causes a developmental disorder (any DD phenotype, even if
      the specific disease differs from the candidate's);
  0 = not gene-DD evidence (functional/non-human only, gene mentioned incidentally, or no
      molecular confirmation).

Inputs: annotated_tiab.csv (pmid, tiab, g2p_lgmde, label) + the augmentation worksheet
(augmentation_candidates_to_annotate.csv with `confirm_positive` filled 1/0/blank). g2p_lgmde
for augmentation rows is rebuilt from the current DDG2P (first 15 columns, HGNC-prefixed).
Output: a merged annotated_tiab-format CSV ready for final_traintest_dataset.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def lgmde_builder(ddg2p_csv: str):
    dd = pd.read_csv(ddg2p_csv)
    dd.columns = [c.strip() for c in dd.columns]
    cols15 = list(dd.columns[:15])
    ddi = dd.drop_duplicates("g2p id").set_index("g2p id", drop=False)

    def build(g2p_id: str) -> str:
        if g2p_id not in ddi.index:
            return g2p_id
        r = ddi.loc[g2p_id]
        out = []
        for c in cols15:
            v = r[c]
            if c == "hgnc id" and pd.notna(v):
                v = f"HGNC:{int(v) if isinstance(v, float) and v == int(v) else v}"
            out.append(str(v))
        return " - ".join(out)
    return build


def collapse_per_pmid(df: pd.DataFrame) -> pd.DataFrame:
    """any 1 -> keep positives, drop 0s ; all 0 -> keep all."""
    parts = []
    for _, grp in df.groupby("pmid", sort=False):
        parts.append(grp[grp["label"] == "1"] if (grp["label"] == "1").any() else grp)
    return pd.concat(parts, ignore_index=True)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotated", required=True, help="existing annotated_tiab.csv")
    ap.add_argument("--augmentation", required=True, help="augmentation worksheet with confirm_positive")
    ap.add_argument("--ddg2p", required=True)
    ap.add_argument("--out", default="revision/external_recall/annotated_tiab_augmented.csv")
    return ap.parse_args()


def main():
    args = parse_args()
    ann = pd.read_csv(args.annotated, dtype=str).fillna("")[["pmid", "tiab", "g2p_lgmde", "label"]]

    aug = pd.read_csv(args.augmentation, dtype=str).fillna("")
    aug["confirm_positive"] = aug["confirm_positive"].str.strip().str.lower()
    aug = aug[aug["confirm_positive"].isin(["0", "1", "yes", "no", "true", "false", "y", "n"])].copy()
    aug["label"] = aug["confirm_positive"].map(
        lambda v: "1" if v in ("1", "yes", "true", "y") else "0")
    build = lgmde_builder(args.ddg2p)
    aug["tiab"] = (aug["title"] + " " + aug["abstract"]).str.strip()
    aug["g2p_lgmde"] = aug["g2p_id"].map(build)
    aug = aug[["pmid", "tiab", "g2p_lgmde", "label"]]

    merged = pd.concat([ann, aug], ignore_index=True)
    before = len(merged)
    collapsed = collapse_per_pmid(merged)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    collapsed.to_csv(args.out, index=False)

    print(f"existing annotated rows: {len(ann)} | augmentation rows added: {len(aug)} "
          f"({int((aug['label'] == '1').sum())} positive, {int((aug['label'] == '0').sum())} negative)")
    print(f"collapse dropped {before - len(collapsed)} conflicting 0-rows (PMIDs that also had a 1)")
    print(f"merged screen set: {len(collapsed)} rows | labels {collapsed['label'].value_counts().to_dict()} "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
