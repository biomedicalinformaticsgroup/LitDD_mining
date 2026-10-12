#!/usr/bin/env python3
"""Attach title-and-abstract text to the annotated CSV, which ships with PMIDs only.

``data/annotated_pmid.csv`` (pmid, g2p_lgmde, label) carries no text, because abstract text can be
under publisher copyright. This step joins it by PMID to a text source and writes the ``tiab``
column the training scripts read, built exactly as the screen builds it (title, one space, abstract).

Text sources (``--tiab_source``):
  * a CSV with ``pmid``, ``title`` and ``abstract`` columns, for example Supplementary Data 1 of the
    paper, or
  * the parquet directory written by ``litdd.pipeline.pubmed_to_parquet``.

Example:
    python -m litdd.training.attach_tiab --annotated_csv data/annotated_pmid.csv \\
        --tiab_source "Supplementary Data 1.csv" --out_csv data/annotated_pmid_tiab.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from litdd.pipeline.screen_io import make_tiab


def load_text(source: str, pmids: set[str]) -> pd.DataFrame:
    """Return ``pmid, tiab`` for ``pmids`` from a CSV or a parquet directory."""
    p = Path(source)
    if p.is_dir():
        import pyarrow.dataset as ds
        t = ds.dataset(str(p), format="parquet").to_table(columns=["pmid", "title", "abstract"])
        df = t.to_pandas()
    else:
        df = pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig",
                         usecols=lambda c: c in {"pmid", "title", "abstract"})
    df["pmid"] = df["pmid"].astype(str)
    df = df[df["pmid"].isin(pmids)].drop_duplicates("pmid", keep="last")
    df["tiab"] = [make_tiab({"title": t, "abstract": a})["tiab"]
                  for t, a in zip(df["title"], df.get("abstract", pd.Series("", index=df.index)))]
    return df[["pmid", "tiab"]]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--annotated_csv", default="data/annotated_pmid.csv")
    ap.add_argument("--tiab_source", required=True, help="CSV (pmid,title,abstract) or pubmed_to_parquet directory")
    ap.add_argument("--out_csv", default="data/annotated_pmid_tiab.csv")
    args = ap.parse_args(argv)

    ann = pd.read_csv(args.annotated_csv, dtype={"pmid": str})
    text = load_text(args.tiab_source, set(ann["pmid"]))
    out = ann.merge(text, on="pmid", how="left")
    missing = out["tiab"].isna() | (out["tiab"] == "")
    if missing.any():
        print(f"[WARN] {int(missing.sum())} rows ({out.loc[missing, 'pmid'].nunique()} PMIDs) have no text "
              "in the source and are dropped", file=sys.stderr)
    out = out[~missing]
    out.to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv}: {len(out)} rows, {out['pmid'].nunique()} PMIDs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
