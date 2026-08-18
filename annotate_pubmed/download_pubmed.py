#!/usr/bin/env python3
"""Download the PubMed/MEDLINE XML corpus (annual baseline + daily updatefiles).

Files already present in the download directory are skipped, so the script is safe to
re-run and is the mechanism behind the daily update CronJob.

IMPORTANT — the annual baseline supersedes everything before it. NCBI reissues the whole
baseline each December under a NEW year prefix (``pubmed25n*`` -> ``pubmed26n*``). The
skip-if-present check is keyed on *filename*, so pointing this script at a directory that
already holds a previous year's baseline downloads the new baseline alongside the old one
and leaves BOTH in place. ``pubmed_to_parquet.py`` then converts both and every pre-existing
record is duplicated, silently inflating every downstream count.

So:
  * moving to a new baseline year  -> use a FRESH ``--download_dir``;
  * topping up within a baseline year -> reuse the directory with ``--updates_only``.

``--check_prefix`` (default on) refuses to mix baseline years in one directory.

Examples
--------
    # Fresh 2026 corpus
    python annotate_pubmed/download_pubmed.py --download_dir data/pubmed_download_2026

    # Daily top-up within the same baseline year
    python annotate_pubmed/download_pubmed.py --download_dir data/pubmed_download_2026 \
        --updates_only
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import requests
from lxml import html

PUBMED_BASELINE = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
PUBMED_UPDATE = "https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/"

# pubmed<YY>n<NNNN>.xml.gz -> captures the two-digit baseline year
PREFIX_RE = re.compile(r"pubmed(\d{2})n\d+\.xml\.gz$")


def get_file_links(base_url: str) -> list[str]:
    """Return absolute URLs of every .xml.gz (excluding .md5) listed at base_url."""
    response = requests.get(base_url, timeout=60)
    if response.status_code != 200:
        print(f"Failed to fetch data from {base_url} (HTTP {response.status_code})")
        return []

    tree = html.fromstring(response.text)
    xpath = '//a[contains(@href, ".xml.gz") and not(contains(@href, ".md5"))]/@href'
    return [base_url + link for link in tree.xpath(xpath)]


def baseline_years(names) -> set[str]:
    """Two-digit baseline years present in an iterable of filenames."""
    return {m.group(1) for m in (PREFIX_RE.search(n) for n in names) if m}


def download_one(file_url: str, download_dir: str) -> tuple[str, bool]:
    """Fetch one file unless it is already present. Returns (name, downloaded?)."""
    file_name = file_url.rsplit("/", 1)[-1]
    local_path = os.path.join(download_dir, file_name)
    if os.path.exists(local_path):
        return file_name, False
    # -q to keep the log readable; wget resumes/overwrites partials itself.
    subprocess.check_call(["wget", "-q", "-P", download_dir, file_url])
    return file_name, True


def download_files(file_links, download_dir: str, workers: int) -> int:
    """Download all missing files, up to `workers` at a time. Returns the count fetched."""
    fetched = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for name, did in pool.map(lambda u: download_one(u, download_dir), file_links):
            if did:
                fetched += 1
                print(f"Downloaded: {name}", flush=True)
    return fetched


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--download_dir", required=True,
                   help="Destination for raw XML; 'raw_download_files' is created inside it")
    p.add_argument("--updates_only", action="store_true",
                   help="Fetch only /updatefiles/ (top up within the current baseline year)")
    p.add_argument("--baseline_only", action="store_true",
                   help="Fetch only /baseline/")
    # NCBI asks for <=3 concurrent connections during US business hours.
    p.add_argument("--workers", type=int, default=4,
                   help="Concurrent downloads (default 4; keep <=3 during US business hours)")
    p.add_argument("--no_check_prefix", dest="check_prefix", action="store_false",
                   help="Allow mixing baseline years in one directory (NOT recommended)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.updates_only and args.baseline_only:
        print("--updates_only and --baseline_only are mutually exclusive", file=sys.stderr)
        return 2

    download_dir = os.path.join(args.download_dir, "raw_download_files")
    os.makedirs(download_dir, exist_ok=True)

    all_files: list[str] = []
    if not args.updates_only:
        all_files += get_file_links(PUBMED_BASELINE)
    if not args.baseline_only:
        all_files += get_file_links(PUBMED_UPDATE)
    if not all_files:
        print("No files listed by the FTP index — aborting.", file=sys.stderr)
        return 1

    existing = os.listdir(download_dir)
    have, want = baseline_years(existing), baseline_years(f.rsplit("/", 1)[-1] for f in all_files)
    if args.check_prefix and len(have | want) > 1:
        print(
            f"ERROR: baseline-year mismatch in {download_dir}.\n"
            f"  already present: {sorted(have) or '(empty)'}\n"
            f"  remote offers  : {sorted(want)}\n"
            "NCBI's annual baseline supersedes the previous year, and mixing years here\n"
            "would duplicate every record downstream. Use a fresh --download_dir for the\n"
            "new baseline, or --updates_only to top up the existing year.\n"
            "Override with --no_check_prefix only if you know what you are doing.",
            file=sys.stderr,
        )
        return 1

    print(f"Listed {len(all_files)} remote file(s); {len(existing)} already in {download_dir}")
    fetched = download_files(all_files, download_dir, args.workers)
    print(f"Done. Fetched {fetched} new file(s); {len(os.listdir(download_dir))} total.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
