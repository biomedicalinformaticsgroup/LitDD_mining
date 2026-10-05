#!/usr/bin/env python3
"""Convert downloaded PubMed/MEDLINE XML into one parquet shard per XML file.

Reads ``<download_dir>/raw_download_files/*.xml.gz`` and writes
``<download_dir>/parquet_download_files/<name>.parquet`` with the columns emitted by
``pubmed_parser.parse_medline_xml`` and ``pubdate`` reduced to a four-digit year. A shard
whose output already exists is skipped, so the job is restartable and works as a daily
top-up alongside ``download_pubmed.py``. Conversion is CPU-bound and parallel across
``--workers`` processes, or across pods with ``--shard`` and ``--num_shards``.

``pubmed_parser`` emits no row for ``<DeleteCitation>`` entries, so withdrawn PMIDs are
extracted from the raw XML by ``extract_deleted_pmids.py`` and removed, together with the
PMIDs that updatefiles reissue, by ``dedupe_pmids.py`` after the whole corpus is converted.

Example
-------
    python -m litdd.pipeline.pubmed_to_parquet --download_dir data/pubmed_download --workers 16
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from glob import glob

import pandas as pd
import pubmed_parser as pp

logger = logging.getLogger("litdd.pubmed_to_parquet")


def parse_pubdate_year(value) -> int:
    """Return the four-digit year of a MEDLINE pubdate string, or 0 when it cannot be parsed."""
    try:
        return int(str(value).split("-")[0])
    except (ValueError, TypeError, AttributeError):
        return 0


def process_file_to_parquet(xml_file: str, output_directory: str) -> tuple[str, str]:
    """Convert one ``.xml.gz`` file to parquet. Returns (xml_file, status)."""
    base_name = os.path.splitext(os.path.splitext(os.path.basename(xml_file))[0])[0]
    output_file = os.path.join(output_directory, f"{base_name}.parquet")

    if os.path.exists(output_file):
        return xml_file, "skipped"

    try:
        docs = pp.parse_medline_xml(xml_file, year_info_only=False)
        df = pd.DataFrame(list(docs))
        # An unparseable date becomes 0, which the screen's year filter excludes.
        df["pubdate"] = [parse_pubdate_year(v) for v in df["pubdate"]]
        df.to_parquet(output_file, engine="pyarrow", index=False)
        return xml_file, f"ok ({len(df)} rows)"
    except Exception as e:
        # One failed shard is recorded in a marker file; the other shards continue.
        error_info = str(e) + "\n" + traceback.format_exc()
        marker = os.path.join(output_directory, f"BAD_DOWNLOAD_{base_name}.txt")
        with open(marker, "w") as f:
            f.write(error_info)
        return xml_file, f"FAILED (logged to {marker})"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line interface."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--download_dir", required=True,
                   help="Directory holding raw_download_files/; parquet_download_files/ is written inside it")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel conversion processes (default 1)")
    p.add_argument("--shard", type=int, default=0, help="Shard index for splitting across pods")
    p.add_argument("--num_shards", type=int, default=1, help="Total shards")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    download_dir = os.path.join(args.download_dir, "raw_download_files")
    output_dir = os.path.join(args.download_dir, "parquet_download_files")
    os.makedirs(output_dir, exist_ok=True)

    xml_files = sorted(glob(os.path.join(download_dir, "*.xml.gz")))
    if args.num_shards > 1:
        xml_files = [f for i, f in enumerate(xml_files) if i % args.num_shards == args.shard]
    logger.info("[shard %d/%d] %d XML file(s) to consider", args.shard, args.num_shards, len(xml_files))

    failures = 0
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = pool.map(process_file_to_parquet, xml_files,
                               [output_dir] * len(xml_files))
            for xml_file, status in results:
                logger.info("%s: %s", os.path.basename(xml_file), status)
                failures += status.startswith("FAILED")
    else:
        for xml_file in xml_files:
            _, status = process_file_to_parquet(xml_file, output_dir)
            logger.info("%s: %s", os.path.basename(xml_file), status)
            failures += status.startswith("FAILED")

    logger.info("Done. %d failure(s).", failures)
    # A partial conversion exits non-zero so the next stage does not start on it.
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
