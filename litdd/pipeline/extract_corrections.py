#!/usr/bin/env python3
"""Extract ``<CommentsCorrections>`` links from the raw MEDLINE XML.

Reads every ``*.xml.gz`` in ``--raw_dir`` and writes ``--out``, a CSV with columns
``pmid,reftype`` holding one row per ``RefType`` attribute found inside a ``<PubmedArticle>``
record, attributed to that record's own PMID. ``dedupe_pmids.py`` reads the CSV to exclude
retracted papers and earlier versions of republished ones.

MEDLINE marks retractions and corrections in two places. ``<PublicationType>`` holds MeSH
descriptors (``D016441:Retracted Publication``, ``D016425:Published Erratum``), which
``pubmed_parser`` exposes as ``publication_types``. ``<CommentsCorrectionsList>`` holds
``RefType`` links (``RetractionIn``, ``ErratumIn``, ``ExpressionOfConcernIn``), which are not
MeSH terms and do not appear in the parsed output; this script recovers them. Retracted
articles and their notices are caught by publication type; the RefType pass adds papers with
an expression of concern and earlier versions of republished articles, which no
publication type expresses.

The ``In``/``Of`` and ``In``/``For`` suffixes point in opposite directions:

    RetractionIn            this paper was retracted
    RetractionOf            this record is the retraction notice
    ErratumIn               this paper has a correction
    ErratumFor              this record is the erratum notice
    ExpressionOfConcernIn   a concern was raised about this paper
    ExpressionOfConcernFor  this record is the concern notice
    UpdateIn / UpdateOf     earlier / later version
    CommentIn / CommentOn   commentary, not a correction

Usage
-----
    python -m litdd.pipeline.extract_corrections \\
        --raw_dir data/pubmed_download/raw_download_files \\
        --out     data/pubmed_download/corrections.csv
"""
from __future__ import annotations

import argparse
import csv
import gzip
import logging
import os
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from glob import glob

logger = logging.getLogger("litdd.extract_corrections")

ART_OPEN = b"<PubmedArticle>"
ART_CLOSE = b"</PubmedArticle>"
PMID_RE = re.compile(rb"<PMID[^>]*>(\d+)</PMID>")
REFTYPE_RE = re.compile(rb'<CommentsCorrections RefType="([A-Za-z]+)"')


def scan_file(path: str) -> list[tuple[str, str]]:
    """Return (pmid, reftype) pairs for one ``.xml.gz`` file.

    The record's own PMID is the first ``<PMID>`` inside the ``<PubmedArticle>`` element;
    PMIDs inside ``<CommentsCorrections>`` refer to the linked paper and are not used.
    """
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
        logger.warning("%s: %s", os.path.basename(path), e)
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line interface."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--raw_dir", required=True, help="Directory of pubmed*.xml.gz files")
    p.add_argument("--out", required=True, help="Output CSV with columns pmid,reftype")
    p.add_argument("--workers", type=int, default=4, help="Parallel readers (default 4)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    files = sorted(glob(os.path.join(args.raw_dir, "*.xml.gz")))
    if not files:
        logger.error("No .xml.gz in %s", args.raw_dir)
        return 1
    logger.info("Scanning %d file(s) with %d workers", len(files), args.workers)

    # Scan files in parallel and stream the links into the CSV as each file completes.
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
                    logger.info("  %d/%d files, %d links", i, len(files), seen)

    logger.info("Wrote %s: %d (pmid, reftype) link(s)", args.out, seen)
    logger.info("RefType totals:")
    for rt, n in tally.most_common():
        logger.info("  %-26s %9d", rt, n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
