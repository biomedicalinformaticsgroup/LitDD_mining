#!/usr/bin/env python3
"""Extract withdrawn PMIDs from ``<DeleteCitation>`` blocks in the raw MEDLINE XML.

WHY THIS EXISTS
---------------
``pubmed_parser.parse_medline_xml()`` exposes a ``delete`` field, and the pipeline relied on
it to drop papers PubMed has withdrawn. **It never fires.** Measured on the full 2026 corpus:
``delete == True`` for 0 of 45,056,462 records, in both the baseline (39,994,988) and the
updatefiles (5,061,474) — yet the raw updatefile XML plainly contains ``<DeleteCitation>``
blocks (e.g. pubmed26n1400 has 46 PMIDs, n1450 has 30, n1500 has 32).

The cause is structural: ``<DeleteCitation>`` is a sibling of ``<PubmedArticle>`` holding
bare ``<PMID>`` elements with no article body, so a parser that walks ``PubmedArticle``
records never emits a row for them. There is nothing for a ``delete`` column to be True on.

Consequence if unfixed: 7,408 withdrawn PMIDs in the 2026 updatefiles, of which 5,025 are
still present in the converted corpus and 4,859 are English and would reach the screen.
Retaining retracted papers is a bad look in a paper about literature-curation quality.

NOTE this is distinct from retractions recorded as publication types. Those DO parse
correctly, as MeSH-prefixed strings in ``publication_types`` (2026 corpus counts):
    D016441:Retracted Publication   36,775   the retracted article itself
    D016440:Retraction Notice      ~37,000   the notice announcing it
    D016425:Published Erratum      242,873
    D016432:Expression of Concern    4,937   (approximate; term appears in combination)
Searching for "Retraction of Publication" finds almost nothing — that is not the MeSH term.

Usage
-----
    python litdd/pipeline/extract_deleted_pmids.py \\
        --raw_dir data/pubmed_download_2026/raw_download_files \\
        --out     data/pubmed_download_2026/deleted_pmids.txt
"""
from __future__ import annotations

import argparse
import gzip
import os
import re
from concurrent.futures import ProcessPoolExecutor
from glob import glob

# <PMID Version="1">12345678</PMID> -- capture only the element text. Matching digits
# anywhere in the tag would also pick up Version="1" and inject a bogus PMID of 1.
PMID_RE = re.compile(rb"<PMID[^>]*>(\d+)</PMID>")
OPEN_RE = b"<DeleteCitation>"
CLOSE_RE = b"</DeleteCitation>"


def deleted_in_file(path: str) -> list[str]:
    """PMIDs inside <DeleteCitation> blocks of one .xml.gz."""
    out: list[str] = []
    inside = False
    try:
        with gzip.open(path, "rb") as fh:
            for line in fh:
                if not inside:
                    if OPEN_RE in line:
                        inside = True
                    else:
                        continue
                if inside:
                    out.extend(m.decode() for m in PMID_RE.findall(line))
                    if CLOSE_RE in line:
                        inside = False
    except OSError as e:  # a truncated download must not kill the sweep
        print(f"[warn] {os.path.basename(path)}: {e}")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--raw_dir", required=True, help="Directory of pubmed*.xml.gz files")
    p.add_argument("--out", required=True, help="Output: one PMID per line, sorted")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel readers (default 4; these are gzip-bound)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    files = sorted(glob(os.path.join(args.raw_dir, "*.xml.gz")))
    if not files:
        raise SystemExit(f"No .xml.gz files in {args.raw_dir}")
    print(f"Scanning {len(files)} file(s) for <DeleteCitation> with {args.workers} workers")

    pmids: set[str] = set()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, got in enumerate(ex.map(deleted_in_file, files, chunksize=4), 1):
            pmids.update(got)
            if i % 200 == 0:
                print(f"  {i}/{len(files)} files, {len(pmids)} withdrawn PMIDs so far")

    ordered = sorted(pmids, key=int)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("\n".join(ordered) + ("\n" if ordered else ""))
    print(f"Wrote {args.out}: {len(ordered)} withdrawn PMID(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
