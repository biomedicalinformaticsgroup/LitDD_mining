#!/usr/bin/env python3
"""Measure the gene-mention filter as a *gate*, not a cleanup (R2-C1 / R3.4).

The pipeline currently applies the gene-mention check last, after the cross-encoder has scored
every TIAB against the whole G2P panel. Moving it ahead of the cross-encoder collapses the
candidate set (~2,861 entries -> a handful) and removes ~75% of the GPU critical path -- but it
turns the check into a hard gate: a paper whose gene is not detected is dropped before anything
downstream can recover it.

So the gate's recall becomes the pipeline's ceiling, and this script measures it before the
architecture is committed. It also measures precision, because the goal is a high-precision
final set: dropping records with no detectable gene may *gain* more precision than it costs in
recall, and that trade-off should be read off a measurement rather than assumed either way.

Reported per configuration:
  * recall    -- of truly-positive (pmid, g2p_id) pairs, the fraction the gate keeps
  * precision -- of pairs the gate keeps, the fraction that are truly positive
  * retained  -- absolute number kept, i.e. how much the gate cuts

Configurations:
  none          no gene filter (recall ceiling, precision floor)
  pubtator      PubTator3 symbol annotations only (the current filter's source)
  name          HGNC descriptive-name matching only  (--hgnc)
  pubtator+name both sources unioned (the proposal)

Example
-------
    python litdd/evaluation/gene_filter_recall.py \\
        --annotated_csv "$REF/train_test/annotated_tiab.csv" \\
        --g2p_csv revision/G2P_DD_2026-06-24.csv \\
        --gene2pubtator .../gene2pubtator3 \\
        --gene_info revision/human_gene_info.gz \\
        --hgnc hgnc_complete_set.txt \\
        --out_csv revision/external_recall/gene_filter_recall.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

from litdd.genes import GeneNameMatcher, load_gene_info, load_pubtator_genes  # noqa: E402

G2P_ID_RE = re.compile(r"^(G2P\d+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--annotated_csv", default=None,
                   help="Labelled set (pmid, tiab, g2p_lgmde, label) -> recall AND precision")
    p.add_argument("--truth_csv", default=None,
                   help="Positives-only truth set -> recall only. Accepts pmid + "
                        "(g2p_id|g2p) + (tiab | title+abstract), optional 'source' column "
                        "for a per-source breakdown (premined/HPOA/ClinGen).")
    p.add_argument("--label", default=None, help="Name for this run in the output CSV")
    p.add_argument("--g2p_csv", required=True)
    p.add_argument("--gene2pubtator", required=True,
                   help="gene2pubtator3 (.gz or plain TSV)")
    p.add_argument("--gene_info", required=True, help="NCBI gene_info.gz")
    p.add_argument("--hgnc", default=None,
                   help="hgnc_complete_set.txt for descriptive-name matching (optional)")
    p.add_argument("--out_csv", default=None)
    return p.parse_args()


def load_truth_rows(path: str) -> list[tuple[str, str, str, str, str]]:
    """(pmid, g2p_id, tiab, label='1', source) from a positives-only truth file.

    Column names vary across the truth artefacts (``g2p_id`` vs ``g2p``; a single ``tiab``
    vs separate ``title``/``abstract``), so both layouts are accepted rather than requiring
    the caller to pre-normalise.
    """
    csv.field_size_limit(10**9)
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pmid = str(r.get("pmid", "")).strip()
            gid = str(r.get("g2p_id") or r.get("g2p") or "").strip()
            if not pmid or not gid:
                continue
            tiab = r.get("tiab")
            if not tiab:
                tiab = f"{r.get('title', '') or ''} {r.get('abstract', '') or ''}".strip()
            rows.append((pmid, gid, tiab, "1", (r.get("source") or "all").strip()))
    return rows


def load_g2p_symbols(path: str) -> dict[str, list[str]]:
    """g2p id -> [gene symbol, *previous gene symbols]."""
    mp: dict[str, list[str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gid = (row.get("g2p id") or "").strip()
            if not gid:
                continue
            syms = [(row.get("gene symbol") or "").strip()]
            syms += [s.strip() for s in (row.get("previous gene symbols") or "").split(";")]
            mp[gid] = [s for s in syms if s]
    return mp


def main() -> int:
    args = parse_args()
    g2p_symbols = load_g2p_symbols(args.g2p_csv)
    panel_symbols = {s for syms in g2p_symbols.values() for s in syms}

    if bool(args.annotated_csv) == bool(args.truth_csv):
        raise SystemExit("Pass exactly one of --annotated_csv or --truth_csv")

    rows = []
    csv.field_size_limit(10**9)
    if args.truth_csv:
        rows = load_truth_rows(args.truth_csv)
        kind = "truth pairs (positives only -> recall reported, precision undefined)"
    else:
        with open(args.annotated_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                m = G2P_ID_RE.match((r.get("g2p_lgmde") or "").strip())
                if not m:
                    continue
                rows.append((str(r["pmid"]).strip(), m.group(1), r.get("tiab") or "",
                             str(r.get("label", "")).strip(), "all"))
        kind = "annotated pairs"

    pmids = {p for p, _, _, _, _ in rows}
    print(f"{kind}: {len(rows)}  unique pmids: {len(pmids)}", flush=True)

    print("loading gene_info ...", flush=True)
    gene_info = load_gene_info(args.gene_info)
    print(f"  human genes: {len(gene_info)}", flush=True)

    print("scanning gene2pubtator3 (this reads the whole file) ...", flush=True)
    pub_genes = load_pubtator_genes(args.gene2pubtator, pmids, gene_info)
    print(f"  pmids with >=1 human gene annotation: {len(pub_genes)}", flush=True)

    matcher = None
    if args.hgnc:
        print("building HGNC name dictionary (G2P genes only) ...", flush=True)
        matcher = GeneNameMatcher.from_hgnc(args.hgnc, panel_symbols)
        print(f"  names indexed: {len(matcher.name_to_symbols)} "
              f"families: {len(matcher.family_to_symbols)}", flush=True)

    name_cache: dict[str, set[str]] = {}

    def detected(pmid: str, tiab: str, source: str) -> set[str]:
        got: set[str] = set()
        if source in ("pubtator", "pubtator+name"):
            got |= pub_genes.get(pmid, set())
        if source in ("name", "pubtator+name") and matcher is not None:
            if pmid not in name_cache:
                name_cache[pmid] = matcher.find(tiab)
            got |= name_cache[pmid]
        return got

    configs = ["none", "pubtator"]
    if matcher is not None:
        configs += ["name", "pubtator+name"]

    by_source = sorted({src for *_, src in rows})
    breakdown = by_source if len(by_source) > 1 else []

    out = []
    for cfg in configs:
        buckets: dict[str, list[int]] = {"all": [0, 0, 0]}  # tp, fp, fn
        for pmid, gid, tiab, label, src in rows:
            is_pos = label == "1"
            kept = True if cfg == "none" else bool(
                set(g2p_symbols.get(gid, [])) & detected(pmid, tiab, cfg)
            )
            for key in ("all", src) if breakdown else ("all",):
                b = buckets.setdefault(key, [0, 0, 0])
                if kept and is_pos:
                    b[0] += 1
                elif kept and not is_pos:
                    b[1] += 1
                elif not kept and is_pos:
                    b[2] += 1
        for key, (tp, fp, fn) in buckets.items():
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            # Undefined on positives-only truth sets; reported as empty rather than 1.0.
            precision = round(tp / (tp + fp), 4) if (tp + fp) and fp + fn else ""
            if args.truth_csv:
                precision = ""
            out.append({"run": args.label or "", "source": key, "config": cfg,
                        "retained": tp + fp, "true_pos_kept": tp, "true_pos_lost": fn,
                        "recall": round(recall, 4), "precision": precision})
        tp, fp, fn = buckets["all"]
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        prec = f"{tp / (tp + fp):.4f}" if (tp + fp) and not args.truth_csv else "n/a"
        print(f"  {cfg:14s} retained={tp+fp:6d}  recall={rec:.4f}  precision={prec}"
              f"  positives lost={fn}", flush=True)
        for key in breakdown:
            t, _, n = buckets.get(key, (0, 0, 0))
            r = t / (t + n) if (t + n) else 0.0
            print(f"      {key:20s} recall={r:.4f}  ({t}/{t + n})", flush=True)

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)
        print(f"wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
