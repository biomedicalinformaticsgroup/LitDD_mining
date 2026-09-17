"""Build the stop list of gene aliases that are really disease names.

Why: HGNC and G2P carry historical locus names as "previous symbols" and "alias symbols" for a
gene -- CADASIL for NOTCH3, RTT for MECP2, CMT2A2 for MFN2, SLOS for DHCR7. In text these tokens
name the DISEASE, not the gene. They reach the pipeline three ways, and each one lets an abstract
that never names the gene (so cannot report molecular confirmation in it) acquire that gene's
G2P entries as candidates:

  1. ``--symbol_fallback`` matches G2P ``previous gene symbols`` verbatim;
  2. PubTator3 normalises a disease acronym to the gene and records it as the mention string;
  3. HGNC ``alias_name`` / ``prev_name`` sometimes hold a disease name, which the descriptive
     name matcher then finds.

Criterion (deliberately narrow, so the list is auditable): an alias is listed when it is
EXACTLY a MONDO disease label or EXACT synonym (so the ontology itself uses the string as the
name of a disease), and it is not the official symbol of any G2P gene. Symbols must be >= 3
characters; descriptive names are compared after lower-casing and whitespace normalisation.

Output TSV columns: kind (symbol|name), gene, alias, source, mondo_example.
"""
from __future__ import annotations

import argparse
import csv


def mondo_disease_strings(obo_path: str) -> dict[str, str]:
    """{exact label or EXACT synonym: the term label} for non-obsolete MONDO terms."""
    out: dict[str, str] = {}
    label, names, obsolete = None, [], False

    def flush():
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
    return " ".join(s.lower().split())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--g2p_csv", required=True, help="G2P export whose genes the gate uses")
    ap.add_argument("--all_g2p_csv", default=None,
                    help="all-panel export; its official symbols are never stop-listed")
    ap.add_argument("--hgnc", required=True, help="hgnc_complete_set.txt")
    ap.add_argument("--mondo_obo", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    mondo = mondo_disease_strings(args.mondo_obo)
    mondo_norm = {norm(k): v for k, v in mondo.items()}

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

    def consider_symbol(gene, alias, source):
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
                for a in (row.get(field) or "").strip('"').split("|"):
                    consider_symbol(gene, a.strip(), source)
            for field, source in (("alias_name", "hgnc_alias_name"),
                                  ("prev_name", "hgnc_previous_name")):
                for a in (row.get(field) or "").strip('"').split("|"):
                    a = a.strip().strip('"')
                    if a and norm(a) in mondo_norm:
                        rows.add(("name", gene, a, source, mondo_norm[norm(a)]))

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["kind", "gene", "alias", "source", "mondo_example"])
        for r in sorted(rows):
            w.writerow(r)
    kinds = {k: sum(1 for r in rows if r[0] == k) for k in ("symbol", "name")}
    print(f"wrote {args.out}: {kinds['symbol']} symbols, {kinds['name']} names "
          f"on {len({r[1] for r in rows})} genes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
