#!/usr/bin/env python3
"""Retrieve full text for a set of PMIDs using Cadmus.

Full text is fetched by feeding the LitDD PMID set into Cadmus
(https://github.com/biomedicalinformaticsgroup/cadmus), which assembles full text
from PMC OA, publisher APIs (Wiley, Elsevier) and other sources.

NOTE ON DATA SHARING: the retrieved full text itself is **not** redistributed in this
repository because of publisher permission/licensing requirements. This script lets you
regenerate it from PMIDs so the pipeline is complete and reproducible. Cadmus writes its
results under ``<outdir>/output/`` (consumed downstream by ``get_fulltext_df.py``).

API keys are read from the environment (NCBI_API_KEY, WILEY_API_KEY, ELSEVIER_API_KEY)
or passed explicitly; do not hard-code keys.
"""
import argparse
import os
import pickle
from pathlib import Path

from cadmus import bioscraping, parsed_to_df


def read_pmids(path):
    with open(path, "r") as f:
        return list({line.strip() for line in f if line.strip()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pmids-file", required=True, help="Text file with one PMID per line (LitDD set)")
    ap.add_argument("--outdir", required=True, help="Output dir; Cadmus writes ./output/ under it")
    ap.add_argument("--email", required=True, help="Contact email for NCBI EUtils")
    ap.add_argument("--ncbi-api-key", default=os.getenv("NCBI_API_KEY"))
    ap.add_argument("--wiley-api-key", default=os.getenv("WILEY_API_KEY"))
    ap.add_argument("--elsevier-api-key", default=os.getenv("ELSEVIER_API_KEY"))
    args = ap.parse_args()

    pmids_path = Path(args.pmids_file).expanduser()
    if not pmids_path.is_absolute():
        pmids_path = (Path.cwd() / pmids_path).resolve()

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    # Cadmus writes relative to the working directory.
    os.chdir(outdir)
    # Prepend EDirect if installed in the home dir.
    os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + str(Path.home() / "edirect")

    pmids = read_pmids(pmids_path)
    print(f"Fetching {len(pmids)} PMIDs from {pmids_path}")
    if not pmids:
        print("No PMIDs found. Exiting.")
        return 1

    bioscraping(pmids, args.email, args.ncbi_api_key,
                wiley_api_key=args.wiley_api_key, elsevier_api_key=args.elsevier_api_key)

    content_dir = Path("output") / "retrieved_parsed_files" / "content_text"
    df = parsed_to_df(path=str(content_dir))
    with open(Path("output") / "parsed_df.p", "wb") as fh:
        pickle.dump(df, fh)
    print("Saved parsed dataframe")
    return 0


if __name__ == "__main__":
    import sys
    import traceback
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
