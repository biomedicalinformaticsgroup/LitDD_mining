#!/usr/bin/env python3
"""Convert downloaded PubMed/MEDLINE XML into one parquet shard per XML file.

Idempotent: a shard whose output already exists is skipped, so the job is restartable
and works as a daily top-up alongside ``download_pubmed.py``. Conversion is CPU-bound
(lxml parsing) and embarrassingly parallel — use ``--workers`` (or ``--shard`` /
``--num_shards`` to spread across pods).

Records that PubMed has withdrawn or revised are handled here rather than downstream:
``pubmed_parser`` exposes a ``delete`` flag for ``DeleteCitation`` entries, and updatefiles
reissue PMIDs that already appear in the baseline. ``dedupe_pmids.py`` resolves both once
the whole corpus is converted.

Example
-------
    python annotate_pubmed/pubmed_to_parquet.py \
        --download_dir data/pubmed_download_2026 --workers 16
"""
from __future__ import annotations

import argparse
import os
import traceback
from concurrent.futures import ProcessPoolExecutor
from glob import glob

import pandas as pd
import pubmed_parser as pp


def parse_pubdate_year(value) -> int:
    """MEDLINE pubdate -> four-digit year int; 0 when unparseable."""
    try:
        return int(str(value).split("-")[0])
    except (ValueError, TypeError, AttributeError):
        return 0


def process_file_to_parquet(xml_file: str, output_directory: str) -> tuple[str, str]:
    """Convert one XML.gz to parquet. Returns (xml_file, status)."""
    base_name = os.path.splitext(os.path.splitext(os.path.basename(xml_file))[0])[0]
    output_file = os.path.join(output_directory, f"{base_name}.parquet")

    if os.path.exists(output_file):
        return xml_file, "skipped"

    try:
        docs = pp.parse_medline_xml(xml_file, year_info_only=False)
        df = pd.DataFrame(list(docs))
        # Unparseable dates become 0 rather than raising, so one malformed record cannot
        # cost the whole shard (the screen's pubdate > 1980 filter then excludes them).
        df["pubdate"] = [parse_pubdate_year(v) for v in df["pubdate"]]
        df.to_parquet(output_file, engine="pyarrow", index=False)
        return xml_file, f"ok ({len(df)} rows)"
    except Exception as e:  # noqa: BLE001 - one bad shard must not stop the corpus
        error_info = str(e) + "\n" + traceback.format_exc()
        marker = os.path.join(output_directory, f"BAD_DOWNLOAD_{base_name}.txt")
        with open(marker, "w") as f:
            f.write(error_info)
        return xml_file, f"FAILED (logged to {marker})"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--download_dir", required=True,
                   help="Directory holding raw_download_files/; parquet_download_files/ is written inside it")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel conversion processes (default 1; 16 is a good value on a CPU node)")
    p.add_argument("--shard", type=int, default=0, help="Shard index for splitting across pods")
    p.add_argument("--num_shards", type=int, default=1, help="Total shards")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    download_dir = os.path.join(args.download_dir, "raw_download_files")
    output_dir = os.path.join(args.download_dir, "parquet_download_files")
    os.makedirs(output_dir, exist_ok=True)

    xml_files = sorted(glob(os.path.join(download_dir, "*.xml.gz")))
    if args.num_shards > 1:
        xml_files = [f for i, f in enumerate(xml_files) if i % args.num_shards == args.shard]
    print(f"[shard {args.shard}/{args.num_shards}] {len(xml_files)} XML file(s) to consider")

    failures = 0
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = pool.map(process_file_to_parquet, xml_files,
                               [output_dir] * len(xml_files))
            for xml_file, status in results:
                print(f"{os.path.basename(xml_file)}: {status}", flush=True)
                failures += status.startswith("FAILED")
    else:
        for xml_file in xml_files:
            _, status = process_file_to_parquet(xml_file, output_dir)
            print(f"{os.path.basename(xml_file)}: {status}", flush=True)
            failures += status.startswith("FAILED")

    print(f"Done. {failures} failure(s).")
    # A partial conversion must not look like success to the stage that follows.
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
