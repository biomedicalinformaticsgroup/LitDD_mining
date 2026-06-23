#!/usr/bin/env python3
"""Assemble a full-text dataframe from one or more Cadmus output directories and join
it onto the LitDD TIAB mappings by PMID.

Full text is retrieved beforehand with Cadmus (see run_cadmus.py:
https://github.com/biomedicalinformaticsgroup/cadmus). The full text itself is not
shared in this repo for publisher-permission reasons; this step rebuilds the
`content_text` parquet that extract_hpo.py annotates with HPO terms.

Each Cadmus output directory is expected to contain:
  retrieved_df/retrieved_df2.json.zip          (PMID <-> unique_id metadata)
  retrieved_parsed_files/content_text/*.zip    (parsed full-text per record)
"""
import argparse
import json
import os
import zipfile

import pandas as pd


def parsed_to_df(content_text_dir: str) -> pd.DataFrame:
    """Read every *.zip of parsed content text into a dataframe keyed by file stem."""
    files, content_text = [], []
    for fn in sorted(os.listdir(content_text_dir)):
        if not fn.endswith(".zip"):
            continue
        stem = fn[:-4]
        with zipfile.ZipFile(os.path.join(content_text_dir, fn), "r") as z:
            for name in z.namelist():
                with z.open(name) as f:
                    content_text.append(f.read().decode())
        files.append(stem)
    df = pd.DataFrame(content_text, columns=["content_text"])
    df.index = files
    df["unique_id"] = df.index.astype(str).str.replace(r".[^.]*$", "", regex=True)
    return df.reset_index()


def load_ret2_df(ret_df2_jsonzip_path: str) -> pd.DataFrame:
    """Load the Cadmus retrieved_df2 metadata (PMID <-> unique_id) from its json.zip."""
    with zipfile.ZipFile(ret_df2_jsonzip_path, "r") as z:
        for name in z.namelist():
            with z.open(name) as f:
                data = json.loads(f.read())
    meta = pd.read_json(data, orient="index")
    meta.pmid = meta.pmid.astype(str)
    meta["unique_id"] = meta.index.astype(str)
    return meta.reset_index()


def get_content_df(cadmus_output_dir: str) -> pd.DataFrame:
    """Merge metadata + parsed content for a single Cadmus output dir -> (pmid, content_text)."""
    retdf2 = os.path.join(cadmus_output_dir, "retrieved_df", "retrieved_df2.json.zip")
    content_dir = os.path.join(cadmus_output_dir, "retrieved_parsed_files", "content_text")
    parse_df = parsed_to_df(content_dir)
    meta_df = load_ret2_df(retdf2)
    merged = meta_df[["pmid", "unique_id"]].merge(parse_df, on="unique_id", how="inner")
    return merged[["pmid", "content_text"]]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiab-mappings", required=True,
                    help="Parquet of LitDD TIAB mappings (must have a `pmid` column)")
    ap.add_argument("--cadmus-output", nargs="+", required=True,
                    help="One or more Cadmus output dirs (each with retrieved_df/ and retrieved_parsed_files/)")
    ap.add_argument("--out", default="final_mappings_fulltext.parquet")
    return ap.parse_args()


def main():
    args = parse_args()
    cadmus_df = pd.concat([get_content_df(d) for d in args.cadmus_output], ignore_index=True)
    cadmus_df.pmid = cadmus_df.pmid.astype(str)

    mappings = pd.read_parquet(args.tiab_mappings)
    mappings.pmid = mappings.pmid.astype(str)

    final_df = mappings.merge(cadmus_df, on="pmid", how="left")
    final_df.to_parquet(args.out)
    print(f"Saved {args.out} ({len(final_df)} rows)")


if __name__ == "__main__":
    main()
