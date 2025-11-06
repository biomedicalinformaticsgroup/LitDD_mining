#!/usr/bin/env python3
import os
import glob
import argparse
import polars as pl

PARQUET_COMPRESSION = "zstd"

def build_positive_parquet(processed_dir: str, out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    pattern = os.path.join(processed_dir, "*_bert_processed.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No processed parquet files found in {processed_dir}")
        return

    lf = (
        pl.scan_parquet(pattern)
        .filter(pl.col("bert_predict") == 1)  # keep only positives
    )

    lf.sink_parquet(out_path, compression=PARQUET_COMPRESSION)
    
    try:
        n = pl.scan_parquet(out_path).select(pl.len()).collect().item()
        print(f"Saved {n} rows with bert_predict == 1 to {out_path}")
    except Exception:
        print(f"Wrote positives to {out_path}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=str, default="data/bert_processed")
    ap.add_argument(
        "--out_path",
        type=str,
        default="pubmed_bert_positive.parquet",
    )
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    build_positive_parquet(args.processed_dir, args.out_path)
