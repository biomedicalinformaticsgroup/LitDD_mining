#!/usr/bin/env python3
"""Extract withdrawn PMIDs from ``<DeleteCitation>`` blocks in the raw MEDLINE XML.

Reads every ``*.xml.gz`` in ``--raw_dir`` and writes ``--out``, a text file with one PMID per
line, sorted numerically. ``dedupe_pmids.py`` reads the file to drop withdrawn records.

``<DeleteCitation>`` is a sibling of ``<PubmedArticle>`` in the updatefiles and holds bare
``<PMID>`` elements with no article body. ``pubmed_parser`` walks ``<PubmedArticle>`` records,
so it emits no row for these entries and its ``delete`` column carries no information about
them; the raw XML is the source. Retractions recorded as MeSH publication types are a separate
mechanism and parse normally into ``publication_types``.

Usage
-----
    python -m litdd.pipeline.extract_deleted_pmids \\
        --raw_dir data/pubmed_download/raw_download_files \\
        --out     data/pubmed_download/deleted_pmids.txt
"""
from __future__ import annotations

import argparse
import gzip
import logging
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from glob import glob

logger = logging.getLogger("litdd.extract_deleted_pmids")

# <PMID Version="1">12345678</PMID>: capture the element text only, so the Version attribute
# is never read as a PMID.
PMID_RE = re.compile(rb"<PMID[^>]*>(\d+)</PMID>")
OPEN_RE = b"<DeleteCitation>"
CLOSE_RE = b"</DeleteCitation>"


def deleted_in_file(path: str) -> list[str]:
    """Return the PMIDs inside ``<DeleteCitation>`` blocks of one ``.xml.gz`` file."""
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
    except OSError as e:
        # A truncated download is reported and skipped; the sweep continues.
        logger.warning("%s: %s", os.path.basename(path), e)
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line interface."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--raw_dir", required=True, help="Directory of pubmed*.xml.gz files")
    p.add_argument("--out", required=True, help="Output: one PMID per line, sorted")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel readers (default 4; the work is gzip-bound)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    files = sorted(glob(os.path.join(args.raw_dir, "*.xml.gz")))
    if not files:
        logger.error("No .xml.gz files in %s", args.raw_dir)
        return 1
    logger.info("Scanning %d file(s) for <DeleteCitation> with %d workers", len(files), args.workers)

    # Scan files in parallel and pool the PMIDs.
    pmids: set[str] = set()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, got in enumerate(ex.map(deleted_in_file, files, chunksize=4), 1):
            pmids.update(got)
            if i % 200 == 0:
                logger.info("  %d/%d files, %d withdrawn PMIDs so far", i, len(files), len(pmids))

    ordered = sorted(pmids, key=int)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("\n".join(ordered) + ("\n" if ordered else ""))
    logger.info("Wrote %s: %d withdrawn PMID(s)", args.out, len(ordered))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
