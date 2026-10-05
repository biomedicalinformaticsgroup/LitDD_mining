#!/usr/bin/env python3
"""Download the PubMed/MEDLINE XML corpus (annual baseline and daily updatefiles).

Reads the NCBI FTP directory listings for ``/pubmed/baseline/`` and ``/pubmed/updatefiles/``
and writes every ``*.xml.gz`` they list into ``<download_dir>/raw_download_files/``. Files
already present are skipped, so the script can be re-run to top up a corpus.

NCBI reissues the whole baseline each December under a new year prefix (``pubmed25n*`` to
``pubmed26n*``). The skip check is keyed on filename, so a directory holding a previous
year's baseline would receive the new baseline alongside it, and ``pubmed_to_parquet.py``
would convert both. ``--check_prefix`` (default on) refuses to mix baseline years in one
directory: use a fresh ``--download_dir`` for a new baseline year and ``--updates_only`` to
top up within a year.

Examples
--------
    # New corpus
    python -m litdd.pipeline.download_pubmed --download_dir data/pubmed_download

    # Daily top-up within the same baseline year
    python -m litdd.pipeline.download_pubmed --download_dir data/pubmed_download --updates_only
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

import requests
from lxml import html

logger = logging.getLogger("litdd.download_pubmed")

PUBMED_BASELINE = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
PUBMED_UPDATE = "https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/"

# pubmed<YY>n<NNNN>.xml.gz: captures the two-digit baseline year.
PREFIX_RE = re.compile(r"pubmed(\d{2})n\d+\.xml\.gz$")


def get_file_links(base_url: str) -> list[str]:
    """Return absolute URLs of every ``.xml.gz`` (excluding ``.md5``) listed at ``base_url``."""
    response = requests.get(base_url, timeout=60)
    if response.status_code != 200:
        logger.error("Failed to fetch data from %s (HTTP %d)", base_url, response.status_code)
        return []

    tree = html.fromstring(response.text)
    xpath = '//a[contains(@href, ".xml.gz") and not(contains(@href, ".md5"))]/@href'
    return [base_url + link for link in tree.xpath(xpath)]


def baseline_years(names: Iterable[str]) -> set[str]:
    """Return the two-digit baseline years present in an iterable of filenames."""
    return {m.group(1) for m in (PREFIX_RE.search(n) for n in names) if m}


def download_one(file_url: str, download_dir: str) -> tuple[str, bool]:
    """Fetch one file unless it is already present. Returns (name, downloaded)."""
    file_name = file_url.rsplit("/", 1)[-1]
    local_path = os.path.join(download_dir, file_name)
    if os.path.exists(local_path):
        return file_name, False
    # wget resumes or overwrites partial files itself; -q keeps the log readable.
    subprocess.check_call(["wget", "-q", "-P", download_dir, file_url])
    return file_name, True


def download_files(file_links: list[str], download_dir: str, workers: int) -> int:
    """Download all missing files, up to ``workers`` at a time. Returns the count fetched."""
    fetched = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for name, did in pool.map(lambda u: download_one(u, download_dir), file_links):
            if did:
                fetched += 1
                logger.info("Downloaded: %s", name)
    return fetched


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line interface."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--download_dir", required=True,
                   help="Destination for raw XML; 'raw_download_files' is created inside it")
    p.add_argument("--updates_only", action="store_true",
                   help="Fetch only /updatefiles/ (top up within the current baseline year)")
    p.add_argument("--baseline_only", action="store_true",
                   help="Fetch only /baseline/")
    # NCBI asks for at most three concurrent connections during US business hours.
    p.add_argument("--workers", type=int, default=3,
                   help="Concurrent downloads (default 3, the NCBI limit during US business hours)")
    p.add_argument("--no_check_prefix", dest="check_prefix", action="store_false",
                   help="Allow mixing baseline years in one directory")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    if args.updates_only and args.baseline_only:
        logger.error("--updates_only and --baseline_only are mutually exclusive")
        return 2

    download_dir = os.path.join(args.download_dir, "raw_download_files")
    os.makedirs(download_dir, exist_ok=True)

    all_files: list[str] = []
    if not args.updates_only:
        all_files += get_file_links(PUBMED_BASELINE)
    if not args.baseline_only:
        all_files += get_file_links(PUBMED_UPDATE)
    if not all_files:
        logger.error("No files listed by the FTP index; aborting.")
        return 1

    # Refuse to place two baseline years in the same directory.
    existing = os.listdir(download_dir)
    have, want = baseline_years(existing), baseline_years(f.rsplit("/", 1)[-1] for f in all_files)
    if args.check_prefix and len(have | want) > 1:
        logger.error(
            "baseline-year mismatch in %s.\n"
            "  already present: %s\n"
            "  remote offers  : %s\n"
            "The annual baseline supersedes the previous year, and mixing years here would\n"
            "duplicate every record downstream. Use a fresh --download_dir for the new\n"
            "baseline, or --updates_only to top up the existing year. --no_check_prefix\n"
            "disables this check.",
            download_dir, sorted(have) or "(empty)", sorted(want),
        )
        return 1

    logger.info("Listed %d remote file(s); %d already in %s", len(all_files), len(existing), download_dir)
    fetched = download_files(all_files, download_dir, args.workers)
    logger.info("Done. Fetched %d new file(s); %d total.", fetched, len(os.listdir(download_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
