#!/usr/bin/env python3
"""Build a gene-conditioned training set for the screen: one example per (abstract, candidate gene).

Reads the annotated CSV (``--annotated``: ``pmid, tiab, g2p_lgmde, label``; the gene is field 1
of ``g2p_lgmde``), optionally an annotation worksheet (``--augmentation``: confirmed rows become
examples, ``--use_unconfirmed`` labels every row positive), the G2P DD CSV (``--ddg2p``, for
previous gene symbols) and optionally NCBI ``gene_info.gz`` (``--gene_info``, for approved full
names).

``--variant`` chooses the conditioning input:

  symbol        the gene symbol
  symbol_names  symbol ; previous symbols ; approved full name
  tiab_tag      the symbol, with the abstract's first alias or full-name mention followed by
                "(symbol)" when the symbol itself is absent

Every row carries ``source`` (annotated or premined_aug), ``evidence_type`` (molecular_human
when the text matches both a functional and a human-subject pattern, else other) and ``fold``
(heldout when the gene falls in fold 0 of 5, else train). Writes
``<out_dir>/gene_conditioned_dataset_<variant>.parquet`` and ``<out_dir>/train_pmids_exclude.csv``
(every PMID in the set, for the recall harness's ``--exclude_pmids``), and prints the counts
by source, fold and evidence type.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from litdd.training.screen_common import fold_name, gene_fullnames, parse_confirm_positive

HUMAN = re.compile(r"\b(patient|proband|clinical|affected|individual|famil|presented with|diagnos|"
                   r"phenotyp|congenital|years? old)", re.I)
FUNC = re.compile(r"\b(in vitro|enzyme activit|recombinant|expression (vector|construct|of)|reporter|"
                  r"transfect|biochemical|crystal structure|protein (structure|stabilit|folding)|"
                  r"fibroblast|cell line|mRNA|cDNA)", re.I)


def cond_string(variant: str, symbol: str, prevs: set[str], fullname: str) -> str:
    """Conditioning text for ``symbol``: the symbol alone, or symbol, previous symbols and full name."""
    if variant == "symbol":
        return symbol
    forms = [symbol] + sorted(prevs) + ([fullname] if fullname else [])
    return " ; ".join(dict.fromkeys(f for f in forms if f))


def tag_tiab(text: str, symbol: str, prevs: set[str], fullname: str) -> str:
    """Insert "(symbol)" after the first alias or full-name mention when the symbol is absent from ``text``."""
    if re.search(rf"(?<![A-Za-z0-9]){re.escape(symbol)}(?![A-Za-z0-9])", text):
        return text
    for form in sorted(prevs, key=len, reverse=True) + ([fullname] if fullname else []):
        if not form:
            continue
        m = re.search(rf"(?<![A-Za-z0-9]){re.escape(form)}(?![A-Za-z0-9])", text, re.I)
        if m:
            return text[:m.end()] + f" ({symbol})" + text[m.end():]
    return text


def evidence_type(text: str) -> str:
    """``molecular_human`` when the text matches both FUNC and HUMAN, otherwise ``other``."""
    return "molecular_human" if (FUNC.search(text) and HUMAN.search(text)) else "other"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotated", required=True, help="annotated CSV (pmid, tiab, g2p_lgmde, label)")
    ap.add_argument("--augmentation", default=None, help="annotation worksheet with a confirm_positive column")
    ap.add_argument("--ddg2p", required=True, help="G2P DD CSV, for previous gene symbols")
    ap.add_argument("--gene_info", default=None, help="NCBI gene_info.gz for the approved full name")
    ap.add_argument("--variant", choices=["symbol", "symbol_names", "tiab_tag"], default="symbol_names")
    ap.add_argument("--use_unconfirmed", action="store_true",
                    help="label every worksheet row positive regardless of confirm_positive")
    ap.add_argument("--out_dir", required=True, help="output directory")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    dd = pd.read_csv(args.ddg2p, dtype=str).fillna("")
    dd.columns = [c.strip() for c in dd.columns]
    prev_by_gene: dict[str, set[str]] = {}
    for g, p in zip(dd["gene symbol"], dd["previous gene symbols"]):
        prev_by_gene.setdefault(g.strip(), set()).update(x.strip() for x in p.replace(";", ",").split(",") if x.strip())
    names = gene_fullnames(args.gene_info)

    def make(text: str, symbol: str, label, source: str) -> dict:
        prevs = prev_by_gene.get(symbol, set())
        full = names.get(symbol, "")
        t = tag_tiab(text, symbol, prevs, full) if args.variant == "tiab_tag" else text
        cond = symbol if args.variant == "tiab_tag" else cond_string(args.variant, symbol, prevs, full)
        return {"text": t, "gene": symbol, "gene_cond": cond, "label": int(label),
                "source": source, "evidence_type": evidence_type(text), "fold": fold_name(symbol, 5)}

    rows = []
    ann = pd.read_csv(args.annotated, dtype=str).fillna("")
    for r in ann.itertuples(index=False):
        parts = [x.strip() for x in r.g2p_lgmde.split(" - ")]
        gene = parts[1] if len(parts) > 1 else ""
        if not gene:
            continue
        rows.append({"pmid": str(r.pmid), **make(r.tiab, gene, r.label, "annotated")})

    if args.augmentation:
        aug = pd.read_csv(args.augmentation, dtype=str).fillna("")
        for r in aug.itertuples(index=False):
            label = 1 if args.use_unconfirmed else parse_confirm_positive(getattr(r, "confirm_positive", ""))
            if label is None:
                continue  # blank or unrecognised confirm_positive
            text = f"{r.title} {r.abstract}"
            rows.append({"pmid": str(r.pmid), **make(text, r.gene, label, "premined_aug")})

    ds = pd.DataFrame(rows).drop_duplicates(["pmid", "gene", "source"])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(out / f"gene_conditioned_dataset_{args.variant}.parquet", index=False)
    pd.Series(sorted(set(ds["pmid"]))).to_csv(out / "train_pmids_exclude.csv", index=False, header=["pmid"])

    print(f"variant={args.variant} | examples: {len(ds)} | positives: {int((ds.label==1).sum())} "
          f"| unique TIABs: {ds['pmid'].nunique()}")
    print(ds.groupby("source").agg(n=("label", "size"), pos=("label", "sum")).to_string())
    print("\nfold x source:")
    print(ds.groupby(["fold", "source"]).size().to_string())
    print(f"\nevidence_type of positives: "
          f"{ds[ds.label==1]['evidence_type'].value_counts().to_dict()}")
    print(f"wrote gene_conditioned_dataset_{args.variant}.parquet + train_pmids_exclude.csv -> {out}")


if __name__ == "__main__":
    main()
