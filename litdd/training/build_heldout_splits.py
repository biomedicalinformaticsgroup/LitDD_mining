#!/usr/bin/env python3
"""Build the training set for a stricter held-out split from the released screen's training set.

The released screen was trained on the annotated training abstracts plus the augmentation and
external-truth positives (``--base_ds_dir``), with 20,000 corpus negatives added by
``build_prevalence_ladder.py``. The leakage-control splits in ``final_traintest_dataset.py
--group_col`` hold out whole genes, whole DDG2P entries, or all papers from 2020 onwards. To
evaluate the released recipe under those splits, this script removes from the released training
set every row that belongs to a held-out group of the split's ``ds_test``:

* ``gene``: rows whose thread names a gene present in the test set;
* ``g2p_id``: rows whose thread names a DDG2P entry present in the test set;
* ``time``: rows published after ``--cutoff_year`` and rows whose year is unknown;
* ``tiab``: no group filter.

For every split, rows whose title and abstract text appears in the test set are also removed.
The filtered set is saved with ``save_to_disk`` and is the ``--train_ds_dir`` for
``build_prevalence_ladder.py``. Gene and entry are parsed from ``g2p_lgmde`` exactly as in
``final_traintest_dataset.py``; publication years come from ``--year_sources`` (CSV or parquet
files with ``pmid`` and ``year`` or ``pubdate`` columns) joined through ``--tiab_pmid_sources``
(CSV files with ``pmid`` and ``tiab`` columns), because the training set carries no PMID.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import pandas as pd

from litdd.training.final_traintest_dataset import LGMDE_GENE_FIELD


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_ds_dir", required=True, help="released training set (save_to_disk directory)")
    p.add_argument("--split_dir", required=True, help="directory holding the split's ds_test")
    p.add_argument("--group", required=True, choices=["tiab", "gene", "g2p_id", "time"])
    p.add_argument("--out_dir", required=True, help="filtered training set (save_to_disk directory)")
    p.add_argument("--cutoff_year", type=int, default=2019, help="time split: keep rows published up to this year")
    p.add_argument("--year_sources", nargs="*", default=[], help="pmid,year|pubdate tables (csv or parquet)")
    p.add_argument("--tiab_pmid_sources", nargs="*", default=[], help="pmid,tiab CSVs mapping text to PMID")
    p.add_argument("--panel_csv", nargs="*", default=[],
                   help="G2P panel CSVs ('g2p id', 'gene symbol') used to resolve the gene of rows whose "
                        "thread carries only the entry id (augmentation and external-truth positives)")
    return p.parse_args()


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def thread_parts(ds, entry_gene: dict[str, str] | None = None) -> tuple[list[str], list[str]]:
    """Gene and entry id per row. Rows whose thread is only the entry id (the augmentation and
    external-truth positives) take their gene from ``entry_gene``."""
    genes, entries = [], []
    for thread in ds["g2p_lgmde"]:
        parts = str(thread).split(" - ")
        entry = parts[0].strip() if parts else ""
        gene = parts[LGMDE_GENE_FIELD].strip() if len(parts) > LGMDE_GENE_FIELD else ""
        if not gene and entry_gene:
            gene = entry_gene.get(entry, "")
        entries.append(entry)
        genes.append(gene)
    return genes, entries


def entry_gene_map(panel_csvs: list[str], *datasets) -> dict[str, str]:
    """Entry id -> gene symbol from the panel files and from every full thread in ``datasets``."""
    out: dict[str, str] = {}
    for path in panel_csvs:
        df = pd.read_csv(path, dtype=str, usecols=["g2p id", "gene symbol"]).dropna()
        for e, g in zip(df["g2p id"], df["gene symbol"]):
            out.setdefault(e.strip(), g.strip())
    for ds in datasets:
        for thread in ds["g2p_lgmde"]:
            parts = str(thread).split(" - ")
            if len(parts) > LGMDE_GENE_FIELD and parts[LGMDE_GENE_FIELD].strip():
                out.setdefault(parts[0].strip(), parts[LGMDE_GENE_FIELD].strip())
    return out


def load_years(year_sources: list[str], tiab_pmid_sources: list[str]) -> dict[str, int]:
    """Return normalised tiab -> publication year."""
    pmid_year: dict[str, int] = {}
    for path in year_sources:
        if path.endswith(".parquet"):
            import pyarrow.parquet as pq
            cols = [c for c in ("pmid", "year", "pubdate") if c in pq.read_schema(path).names]
            df = pd.read_parquet(path, columns=cols)
        else:
            df = pd.read_csv(path, dtype=str)
        col = "year" if "year" in df.columns else "pubdate"
        for pmid, val in zip(df["pmid"].astype(str), df[col].astype(str)):
            m = re.match(r"(\d{4})", val)
            if m and pmid not in pmid_year:
                pmid_year[pmid] = int(m.group(1))
    tiab_year: dict[str, int] = {}
    for path in tiab_pmid_sources:
        df = pd.read_csv(path, dtype=str, usecols=["pmid", "tiab"]).dropna()
        for pmid, tiab in zip(df["pmid"], df["tiab"]):
            y = pmid_year.get(str(pmid))
            if y is not None:
                tiab_year.setdefault(norm(tiab), y)
    print(f"years known for {len(pmid_year):,} PMIDs, {len(tiab_year):,} texts", flush=True)
    return tiab_year


def main() -> int:
    a = parse_args()
    from datasets import load_from_disk

    base = load_from_disk(a.base_ds_dir)
    test = load_from_disk(os.path.join(a.split_dir, "ds_test"))
    entry_gene = entry_gene_map(a.panel_csv, base, test)
    base_genes, base_entries = thread_parts(base, entry_gene)
    test_genes, test_entries = thread_parts(test, entry_gene)
    print(f"entry->gene map: {len(entry_gene):,} entries; rows with no resolvable gene: "
          f"{sum(1 for g in base_genes if not g):,} in base, {sum(1 for g in test_genes if not g):,} in test", flush=True)
    test_texts = {norm(t) for t in test["tiab"]}
    n = len(base)
    keep = [True] * n
    reason = {"test text": 0, "held-out group": 0, "unknown year": 0}

    if a.group == "gene":
        held = set(test_genes)
        for i, g in enumerate(base_genes):
            if g in held:
                keep[i] = False
                reason["held-out group"] += 1
            elif not g:                      # gene unknown: cannot be shown leak-free, so drop
                keep[i] = False
                reason["unknown gene"] = reason.get("unknown gene", 0) + 1
    elif a.group == "g2p_id":
        held = set(test_entries)
        for i, e in enumerate(base_entries):
            if e in held:
                keep[i] = False
                reason["held-out group"] += 1
    elif a.group == "time":
        years = load_years(a.year_sources, a.tiab_pmid_sources)
        for i, t in enumerate(base["tiab"]):
            y = years.get(norm(t))
            if y is None:
                keep[i] = False
                reason["unknown year"] += 1
            elif y > a.cutoff_year:
                keep[i] = False
                reason["held-out group"] += 1

    for i, t in enumerate(base["tiab"]):
        if keep[i] and norm(t) in test_texts:
            keep[i] = False
            reason["test text"] += 1

    idx = [i for i in range(n) if keep[i]]
    out = base.select(idx)
    pos = sum(1 for x in out["label"] if int(x) == 1)
    print(f"split {a.group}: base {n:,} rows -> {len(out):,} kept ({pos:,} positive, "
          f"{100 * pos / max(len(out), 1):.1f}%); removed {reason}", flush=True)
    if a.group in ("gene", "g2p_id"):
        left = set(base_genes if a.group == "gene" else base_entries)
        overlap = (set(test_genes if a.group == "gene" else test_entries)
                   & {(base_genes if a.group == "gene" else base_entries)[i] for i in idx})
        print(f"held-out {a.group}s in test: {len(set(test_genes if a.group == 'gene' else test_entries)):,}; "
              f"remaining in train: {len(overlap)} (must be 0); groups in base: {len(left):,}", flush=True)
        if overlap:
            return 1
    out.save_to_disk(a.out_dir)
    print(f"saved {a.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
