"""Build the table of gene aliases and alias names that are MONDO disease strings.

Reads the G2P export (``gene symbol``, ``previous gene symbols``), optionally the all-panel
export (its official symbols are never listed), ``hgnc_complete_set.txt`` and a MONDO OBO
release. Writes a TSV with columns ``kind`` (symbol or name), ``gene``, ``alias``, ``source``
(``g2p_previous_symbol``, ``hgnc_alias_symbol``, ``hgnc_previous_symbol``, ``hgnc_alias_name``,
``hgnc_previous_name``) and ``mondo_example`` (the label of a MONDO term using the string).

A symbol is listed when it is at least three characters, is not the official symbol of any
panel gene and is exactly a MONDO disease label or EXACT synonym (CADASIL for NOTCH3, RTT for
MECP2). A name is listed when it equals a MONDO label or EXACT synonym after lower-casing and
whitespace collapsing.

The checked-in output is ``litdd/pipeline/data/disease_name_aliases.tsv``. Its only consumer is
the context-thread builder ``litdd/pipeline/build_context_threads.py``, which removes these
strings from the previous-symbols line of each G2P entry's thread.

Example::

    python -m litdd.pipeline.build_disease_name_aliases \\
        --g2p_csv data/G2P_DD.csv --all_g2p_csv data/G2P_all.csv \\
        --hgnc data/reference/hgnc_complete_set.txt --mondo_obo data/reference/mondo.obo \\
        --out litdd/pipeline/data/disease_name_aliases.tsv
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys

from litdd.genes import split_pipe

logger = logging.getLogger(__name__)


def mondo_disease_strings(obo_path: str) -> dict[str, str]:
    """{label or EXACT synonym: label of the first term using it} over non-obsolete OBO terms."""
    out: dict[str, str] = {}
    label, names, obsolete = None, [], False

    def flush() -> None:
        if label and not obsolete:
            for n in names:
                out.setdefault(n, label)

    with open(obo_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("[Term]") or line.startswith("[Typedef]"):
                flush()
                label, names, obsolete = None, [], False
            elif line.startswith("name: "):
                label = line[6:].strip()
                names.append(label)
            elif line.startswith("synonym: ") and " EXACT" in line:
                names.append(line.split('"')[1].strip())
            elif line.startswith("is_obsolete: true"):
                obsolete = True
    flush()
    return out


def norm(s: str) -> str:
    """Lower-cased with whitespace collapsed; punctuation is kept, unlike ``genes.normalise_phrase``."""
    return " ".join(s.lower().split())


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--g2p_csv", required=True, help="G2P export whose genes are considered")
    ap.add_argument("--all_g2p_csv", default=None,
                    help="all-panel export; its official symbols are never listed")
    ap.add_argument("--hgnc", required=True, help="hgnc_complete_set.txt")
    ap.add_argument("--mondo_obo", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    mondo = mondo_disease_strings(args.mondo_obo)
    mondo_norm = {norm(k): v for k, v in mondo.items()}

    # Panel genes and their G2P previous symbols.
    panel: dict[str, list[str]] = {}
    with open(args.g2p_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gene = (row.get("gene symbol") or "").strip()
            prev = [s.strip() for s in (row.get("previous gene symbols") or "").split(";")]
            panel.setdefault(gene, []).extend(p for p in prev if p)
    official = set(panel)
    if args.all_g2p_csv:
        with open(args.all_g2p_csv, newline="", encoding="utf-8") as f:
            official |= {(r.get("gene symbol") or "").strip() for r in csv.DictReader(f)}

    rows: set[tuple[str, str, str, str, str]] = set()

    def consider_symbol(gene: str, alias: str, source: str) -> None:
        if len(alias) >= 3 and alias not in official and alias in mondo:
            rows.add(("symbol", gene, alias, source, mondo[alias]))

    for gene, prevs in panel.items():
        for a in prevs:
            consider_symbol(gene, a, "g2p_previous_symbol")

    with open(args.hgnc, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            gene = (row.get("symbol") or "").strip()
            if gene not in panel:
                continue
            for field, source in (("alias_symbol", "hgnc_alias_symbol"),
                                  ("prev_symbol", "hgnc_previous_symbol")):
                for a in split_pipe(row.get(field)):
                    consider_symbol(gene, a, source)
            for field, source in (("alias_name", "hgnc_alias_name"),
                                  ("prev_name", "hgnc_previous_name")):
                for a in split_pipe(row.get(field)):
                    if norm(a) in mondo_norm:
                        rows.add(("name", gene, a, source, mondo_norm[norm(a)]))

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["kind", "gene", "alias", "source", "mondo_example"])
        for r in sorted(rows):
            w.writerow(r)
    kinds = {k: sum(1 for r in rows if r[0] == k) for k in ("symbol", "name")}
    logger.info("wrote %s: %d symbols, %d names on %d genes", args.out, kinds["symbol"], kinds["name"],
                len({r[1] for r in rows}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
