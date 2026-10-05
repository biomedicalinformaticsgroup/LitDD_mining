#!/usr/bin/env python3
"""Merge an annotation worksheet into the screen's annotated set.

Reads ``--annotated`` (``pmid, tiab, g2p_lgmde, label``), the worksheet ``--augmentation``
(``pmid, title, abstract, g2p_id, confirm_positive``) and the G2P DD CSV ``--ddg2p``. Worksheet
rows whose ``confirm_positive`` holds an accepted token (1/0, yes/no, true/false, y/n) become
annotated rows: ``tiab`` is title and abstract joined, and ``g2p_lgmde`` is rebuilt from the
first fifteen G2P columns of the entry (``hgnc id`` prefixed with ``HGNC:``).

The screen classifies an abstract, not an abstract-entry pair, so a PMID is not allowed to
carry both labels. After concatenation the rows are collapsed per PMID: when any row of a PMID
is labelled 1 the rows labelled 0 are dropped; when every row is 0 all rows are kept.

Writes the collapsed set to ``--out`` in the ``annotated`` format, ready for
``final_traintest_dataset.py``, and prints the row counts before and after collapsing.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from litdd.training.screen_common import confirmed_worksheet


def lgmde_builder(ddg2p_csv: str) -> Callable[[str], str]:
    """Return a function mapping a G2P id to its ``g2p_lgmde`` string (first fifteen G2P columns)."""
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
    """Per PMID: keep only the rows labelled "1" when there are any, otherwise keep every row."""
    parts = []
    for _, grp in df.groupby("pmid", sort=False):
        parts.append(grp[grp["label"] == "1"] if (grp["label"] == "1").any() else grp)
    return pd.concat(parts, ignore_index=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotated", required=True, help="annotated CSV (pmid, tiab, g2p_lgmde, label)")
    ap.add_argument("--augmentation", required=True, help="annotation worksheet with a confirm_positive column")
    ap.add_argument("--ddg2p", required=True, help="G2P DD CSV")
    ap.add_argument("--out", required=True, help="merged CSV in the annotated format")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ann = pd.read_csv(args.annotated, dtype=str).fillna("")[["pmid", "tiab", "g2p_lgmde", "label"]]

    aug = confirmed_worksheet(pd.read_csv(args.augmentation, dtype=str).fillna(""))
    aug["label"] = aug["label"].astype(str)
    build = lgmde_builder(args.ddg2p)
    aug["g2p_lgmde"] = aug["g2p_id"].map(build)
    aug = aug[["pmid", "tiab", "g2p_lgmde", "label"]]

    merged = pd.concat([ann, aug], ignore_index=True)
    before = len(merged)
    collapsed = collapse_per_pmid(merged)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    collapsed.to_csv(args.out, index=False)

    print(f"existing annotated rows: {len(ann)} | augmentation rows added: {len(aug)} "
          f"({int((aug['label'] == '1').sum())} positive, {int((aug['label'] == '0').sum())} negative)")
    print(f"collapse dropped {before - len(collapsed)} 0-rows of PMIDs that also had a 1")
    print(f"merged screen set: {len(collapsed)} rows | labels {collapsed['label'].value_counts().to_dict()} "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
