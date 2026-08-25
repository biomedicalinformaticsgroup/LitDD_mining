#!/usr/bin/env python3
"""Extract ``<CommentsCorrections>`` links from raw MEDLINE XML.

WHY THIS EXISTS
---------------
MEDLINE marks retractions, errata and expressions of concern in **two independent places**:

1. ``<PublicationType>`` -- MeSH descriptors (``D016441:Retracted Publication``,
   ``D016425:Published Erratum``, ...). ``pubmed_parser`` exposes these as
   ``publication_types``, and every one of the 83 distinct values observed in the 2026
   corpus is a MeSH descriptor, because MEDLINE draws publication types from a MeSH subset.

2. ``<CommentsCorrectionsList>`` -- ``RefType`` attributes such as ``RetractionIn``,
   ``ErratumIn`` and ``ExpressionOfConcernIn``, which are **not MeSH** and are **not exposed
   by pubmed_parser at all** (no such column exists in the parsed output).

The second is the more reliable retraction marker for the *retracted paper itself*:
``RetractionIn`` says "a retraction notice for this paper exists at PMID X", and it can be
present on records whose ``publication_types`` has not been updated to include
``D016441``. Filtering on publication type alone therefore leaks retracted papers.

Directionality matters -- the ``In``/``Of`` and ``In``/``For`` suffixes are opposites:
    RetractionIn            this paper WAS retracted            -> exclude
    RetractionOf            this IS the retraction notice       -> exclude
    ErratumIn               this paper HAS a correction         -> judgement call
    ErratumFor              this IS the erratum notice          -> exclude
    ExpressionOfConcernIn   concern was raised about this paper -> exclude
    ExpressionOfConcernFor  this IS the concern notice          -> exclude
    UpdateIn / UpdateOf     superseded / superseding version
    CommentIn / CommentOn   commentary, not a correction

Usage
-----
    python litdd/pipeline/extract_corrections.py \\
        --raw_dir data/pubmed_download_2026/raw_download_files \\
        --out     data/pubmed_download_2026/corrections.csv
"""
from __future__ import annotations

import argparse
import csv
import gzip
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from glob import glob

ART_OPEN = b"<PubmedArticle>"
ART_CLOSE = b"</PubmedArticle>"
PMID_RE = re.compile(rb"<PMID[^>]*>(\d+)</PMID>")
REFTYPE_RE = re.compile(rb'<CommentsCorrections RefType="([A-Za-z]+)"')


def scan_file(path: str) -> list[tuple[str, str]]:
    """(pmid, reftype) pairs for one .xml.gz. The article's own PMID is the first
    <PMID> seen inside the record; PMIDs inside CommentsCorrections refer to the
    *other* paper and must not be attributed to this one."""
    out: list[tuple[str, str]] = []
    pmid: str | None = None
    reftypes: list[str] = []
    inside = False
    try:
        with gzip.open(path, "rb") as fh:
            for line in fh:
                if ART_OPEN in line:
                    inside, pmid, reftypes = True, None, []
                if not inside:
                    continue
                if pmid is None:
                    m = PMID_RE.search(line)
                    if m:
                        pmid = m.group(1).decode()
                reftypes.extend(m.decode() for m in REFTYPE_RE.findall(line))
                if ART_CLOSE in line:
                    if pmid and reftypes:
                        out.extend((pmid, rt) for rt in reftypes)
                    inside, pmid, reftypes = False, None, []
    except OSError as e:
        print(f"[warn] {os.path.basename(path)}: {e}")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--raw_dir", required=True)
    p.add_argument("--out", required=True, help="CSV: pmid,reftype")
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    files = sorted(glob(os.path.join(args.raw_dir, "*.xml.gz")))
    if not files:
        raise SystemExit(f"No .xml.gz in {args.raw_dir}")
    print(f"Scanning {len(files)} file(s) with {args.workers} workers")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tally: Counter[str] = Counter()
    seen = 0
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pmid", "reftype"])
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, rows in enumerate(ex.map(scan_file, files, chunksize=4), 1):
                w.writerows(rows)
                tally.update(rt for _, rt in rows)
                seen += len(rows)
                if i % 200 == 0:
                    print(f"  {i}/{len(files)} files, {seen} links")

    print(f"\nWrote {args.out}: {seen} (pmid, reftype) link(s)")
    print("\nRefType totals:")
    for rt, n in tally.most_common():
        print(f"  {rt:26s} {n:>9,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
