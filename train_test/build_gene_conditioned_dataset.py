#!/usr/bin/env python3
"""Build a gene-conditioned training set for the BERT screen (Reviewer 3 R3.4 / R1.3).

Each example is a (TIAB, candidate gene) pair -> "does this abstract give evidence that THIS
gene causes a developmental disorder?" — moving the gene-awareness the cross-encoder already
uses into the screen, so multi-gene/cohort abstracts aren't dropped before a gene-aware stage.

Sources:
  - the annotated set (`annotated_tiab.csv`: pmid, tiab, g2p_lgmde, label) — already in the
    (TIAB, candidate gene-disease) -> label form, incl. TIABs positive for >1 disorder;
  - confirmed molecular-framed positives from `select_augmentation_candidates.py`
    (`confirm_positive` in {1,yes,true}), added as positives — TRAIN fold only.

Conditioning-input variants (--variant), to A/B for recall vs precision:
  symbol       : [SEP] <symbol>
  symbol_names : [SEP] <symbol> ; <previous symbols> ; <approved full name>   (compact; <=~5 extra tokens)
  tiab_tag     : conditioning = <symbol>, and the TIAB is normalised — the in-text mention
                 (alias/full name) is tagged with the canonical symbol so it aligns with the token.

Every row carries `source` (annotated|premined_aug), `evidence_type` (molecular_human|other) and
`fold` (train|heldout) so augmented positives can be up-weighted/ablated and held-out genes kept
out of training. Also writes `train_pmids_exclude.csv` (all training PMIDs) for the recall harness
`--exclude_pmids`, so the external-recall eval cannot memorise trained papers.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import re
from pathlib import Path

import pandas as pd

HUMAN = re.compile(r"\b(patient|proband|clinical|affected|individual|famil|presented with|diagnos|"
                   r"phenotyp|congenital|years? old)", re.I)
FUNC = re.compile(r"\b(in vitro|enzyme activit|recombinant|expression (vector|construct|of)|reporter|"
                  r"transfect|biochemical|crystal structure|protein (structure|stabilit|folding)|"
                  r"fibroblast|cell line|mRNA|cDNA)", re.I)


def fold(gene: str) -> str:
    return "heldout" if int(hashlib.md5(gene.encode()).hexdigest(), 16) % 5 == 0 else "train"


def approved_names(gene_info_gz: str | None) -> dict[str, str]:
    if not gene_info_gz:
        return {}
    out = {}
    with gzip.open(gene_info_gz, "rt") as fh:
        h = {c: i for i, c in enumerate(fh.readline().lstrip("#").rstrip().split("\t"))}
        for ln in fh:
            p = ln.rstrip("\n").split("\t")
            if p[h["description"]] not in ("", "-"):
                out[p[h["Symbol"]]] = p[h["description"]]
    return out


def cond_string(variant, symbol, prevs, fullname):
    if variant == "symbol":
        return symbol
    forms = [symbol] + sorted(prevs) + ([fullname] if fullname else [])
    return " ; ".join(dict.fromkeys(f for f in forms if f))  # symbol_names (also used as base for tiab_tag)


def tag_tiab(text, symbol, prevs, fullname):
    """Insert the canonical symbol next to the first alias/full-name mention if the symbol is absent."""
    if re.search(rf"(?<![A-Za-z0-9]){re.escape(symbol)}(?![A-Za-z0-9])", text):
        return text
    for form in sorted(prevs, key=len, reverse=True) + ([fullname] if fullname else []):
        if not form:
            continue
        m = re.search(rf"(?<![A-Za-z0-9]){re.escape(form)}(?![A-Za-z0-9])", text, re.I)
        if m:
            return text[:m.end()] + f" ({symbol})" + text[m.end():]
    return text


def evidence_type(text):
    return "molecular_human" if (FUNC.search(text) and HUMAN.search(text)) else "other"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotated", required=True, help="annotated_tiab.csv (pmid, tiab, g2p_lgmde, label)")
    ap.add_argument("--augmentation", default=None, help="augmentation_candidates_to_annotate.csv")
    ap.add_argument("--ddg2p", required=True, help="for gene previous symbols")
    ap.add_argument("--gene_info", default=None, help="NCBI gene_info.gz for the approved full name")
    ap.add_argument("--variant", choices=["symbol", "symbol_names", "tiab_tag"], default="symbol_names")
    ap.add_argument("--use_unconfirmed", action="store_true",
                    help="include all augmentation rows as positive (dry run before annotation)")
    ap.add_argument("--out_dir", default="revision/external_recall")
    return ap.parse_args()


def main():
    args = parse_args()
    dd = pd.read_csv(args.ddg2p, dtype=str).fillna("")
    dd.columns = [c.strip() for c in dd.columns]
    prev_by_gene = {}
    for g, p in zip(dd["gene symbol"], dd["previous gene symbols"]):
        prev_by_gene.setdefault(g.strip(), set()).update(x.strip() for x in p.replace(";", ",").split(",") if x.strip())
    names = approved_names(args.gene_info)

    def make(text, symbol, label, source):
        prevs = prev_by_gene.get(symbol, set())
        full = names.get(symbol, "")
        t = tag_tiab(text, symbol, prevs, full) if args.variant == "tiab_tag" else text
        cond = symbol if args.variant == "tiab_tag" else cond_string(args.variant, symbol, prevs, full)
        return {"text": t, "gene": symbol, "gene_cond": cond, "label": int(label),
                "source": source, "evidence_type": evidence_type(text), "fold": fold(symbol)}

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
            cp = str(getattr(r, "confirm_positive", "")).strip().lower()
            if args.use_unconfirmed:
                label = 1
            elif cp in ("1", "yes", "true", "y"):
                label = 1
            elif cp in ("0", "no", "false", "n"):  # reviewer-confirmed negative -> hard negative
                label = 0
            else:
                continue  # blank = not yet annotated / skip
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
    print("\nfold x source (augmentation must be train-only):")
    print(ds.groupby(["fold", "source"]).size().to_string())
    print(f"\nevidence_type of positives: "
          f"{ds[ds.label==1]['evidence_type'].value_counts().to_dict()}")
    print(f"wrote gene_conditioned_dataset_{args.variant}.parquet + train_pmids_exclude.csv -> {out}")


if __name__ == "__main__":
    main()
