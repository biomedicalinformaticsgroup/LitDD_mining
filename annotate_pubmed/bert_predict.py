#!/usr/bin/env python3
import os
import gc
import argparse
import traceback
from typing import Dict, Any, Optional, Tuple, List

import torch
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Enable TF32 matmul on Ampere/Hopper GPUs
if torch.cuda.is_available():
    # Optional: only enable on devices with SM >= 8.0 (Ampere+)
    major, _ = torch.cuda.get_device_capability(0)
    if major >= 8:
        torch.set_float32_matmul_precision('high')
        # Optional: also allow TF32 for convs 
        torch.backends.cudnn.allow_tf32 = True


BERT_MODEL_PATH = "path_to_lit_dd_BERT"

INPUT_DIR = "path_to_pubmed_download/parquet_download_files"
PROCESSED_DIR = "bert_processed"

ROW_BATCH_SIZE = 8192     # CPU-side batch (streaming)
PRED_BATCH_SIZE = 32      # GPU batch size
MAX_LENGTH = 512
PARQUET_COMPRESSION = "zstd"  # change to "snappy" for faster IO (bigger files)
SKIP_IF_EXISTS = True

def get_device(device_str: Optional[str] = None) -> str:
    if device_str:
        return device_str
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_model_and_tokenizer(device_str: Optional[str] = None) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification, str]:
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
    model.eval()
    device = get_device(device_str)
    model.to(device)

    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    return tokenizer, model, device

def safe_pubdate_gt_1980(x: Dict[str, Any]) -> bool:
    try:
        pd = x.get("pubdate", None)
        pd = int(pd) if pd is not None else -1
    except Exception:
        pd = -1
    return (x.get("languages") == "eng") and (pd > 1980)

def make_tiab(x: Dict[str, Any]) -> Dict[str, Any]:
    title = x.get("title", "") or ""
    abstract = x.get("abstract", "") or ""
    x["tiab"] = f"{title} {abstract}".strip()
    return x

@torch.inference_mode()
def predict_batch(tokenizer, model, device, texts: List[str], pred_bs: int = PRED_BATCH_SIZE) -> List[int]:
    preds: List[int] = []
    amp_dtype = torch.bfloat16 if (device.startswith("cuda") and torch.cuda.is_bf16_supported()) else torch.float16
    for i in range(0, len(texts), pred_bs):
        chunk = texts[i:i + pred_bs]
        enc = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        for k in enc:
            enc[k] = enc[k].pin_memory()
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.startswith("cuda")):
            logits = model(**enc).logits
        preds.extend(torch.argmax(logits, dim=-1).detach().cpu().tolist())
        del enc, logits
    return preds


def get_output_schema(parquet_path: str) -> pa.Schema:
    base = pq.read_schema(parquet_path)
    fields = list(base)
    if "tiab" not in base.names:
        fields.append(pa.field("tiab", pa.string()))
    if "bert_predict" not in base.names:
        fields.append(pa.field("bert_predict", pa.int64()))
    return pa.schema(fields)

def table_from_batch_with_schema(batch: Dict[str, List[Any]], preds: List[int], schema: pa.Schema) -> pa.Table:
    if len(preds) > 0:
        n = len(preds)
    elif batch:
        first_key = next(iter(batch))
        n = len(batch[first_key])
    else:
        n = 0

    columns = {}
    for field in schema:
        name = field.name
        typ = field.type
        if name == "bert_predict":
            arr = pa.array(preds, type=pa.int64())
        elif name == "tiab":
            vals = batch.get("tiab", [""] * n)
            arr = pa.array(vals, type=pa.string())
        else:
            if name in batch:
                vals = batch[name]
                arr = pa.array(vals, type=typ)
            else:
                arr = pa.nulls(n, type=typ)
        columns[name] = arr

    table = pa.Table.from_arrays([columns[f.name] for f in schema], schema=schema)
    return table

def process_one_parquet(
    parquet_path: str,
    out_dir: str,
    tokenizer,
    model,
    device: str,
) -> bool:
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(parquet_path)
    stem = os.path.splitext(base)[0]
    out_path = os.path.join(out_dir, f"{stem}_bert_processed.parquet")

    if SKIP_IF_EXISTS and os.path.exists(out_path):
        print(f"Skipping (already exists): {out_path}")
        return True

    try:
        ds = load_dataset("parquet", data_files=parquet_path, split="train", streaming=True)
        ds = ds.filter(safe_pubdate_gt_1980)
        ds = ds.map(make_tiab)
        ds = ds.batch(ROW_BATCH_SIZE)
    except Exception:
        print(f"[ERROR] Failed to open or prepare dataset for: {parquet_path}")
        traceback.print_exc()
        return False

    try:
        out_schema = get_output_schema(parquet_path)
    except Exception:
        print(f"[ERROR] Failed to read schema from: {parquet_path}")
        traceback.print_exc()
        return False

    writer = None
    total_rows = 0
    file_failed = False

    try:
        for batch in ds:
            try:
                texts = batch.get("tiab", [])
                if not texts:
                    continue
                preds = predict_batch(tokenizer, model, device, texts)
                table = table_from_batch_with_schema(batch, preds, out_schema)
                if writer is None:
                    writer = pq.ParquetWriter(out_path, schema=out_schema, compression=PARQUET_COMPRESSION)
                writer.write_table(table)
                total_rows += table.num_rows

                del batch, table, preds, texts
                gc.collect()
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
            except Exception:
                file_failed = True
                print(f"[ERROR] Failed processing a batch in: {parquet_path}")
                traceback.print_exc()
                break
    except Exception:
        file_failed = True
        print(f"[ERROR] Iteration over dataset failed for: {parquet_path}")
        traceback.print_exc()
    finally:
        try:
            if writer is not None:
                writer.close()
        except Exception:
            print(f"[WARN] Failed to close writer for: {out_path}")
            traceback.print_exc()

    if file_failed:
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
                print(f"[CLEANUP] Removed partial output: {out_path}")
            except Exception:
                print(f"[WARN] Failed to remove partial output: {out_path}")
                traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False

    if total_rows > 0:
        print(f"Wrote {total_rows} rows to {out_path}")
    else:
        print(f"No eligible rows in {parquet_path}; no output written.")
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
                print(f"[CLEANUP] Removed empty output: {out_path}")
            except Exception:
                print(f"[WARN] Failed to remove empty output: {out_path}")
                traceback.print_exc()

    return True

def process_all_parquets(
    input_dir: str,
    processed_dir: str,
    shard: int = 0,
    num_shards: int = 1,
    device: Optional[str] = None,
    fail_fast: bool = False,
):
    tokenizer, model, device = load_model_and_tokenizer(device)
    files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".parquet")])
    if not files:
        print(f"No parquet files found in {input_dir}")
        return

    files = [p for i, p in enumerate(files) if i % max(num_shards, 1) == shard]

    for path in files:
        print(f"[shard {shard}/{num_shards}] Processing: {path}")
        try:
            ok = process_one_parquet(path, processed_dir, tokenizer, model, device)
            if not ok:
                if fail_fast:
                    raise RuntimeError(f"Stopping due to error on file: {path}")
                else:
                    print(f"[SKIP] Skipping file due to error: {path}")
                    continue
        except Exception:
            print(f"[ERROR] Unhandled exception while processing: {path}")
            traceback.print_exc()
            if fail_fast:
                raise

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0, help="Shard index for file list")
    ap.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    ap.add_argument("--device", type=str, default=None, help="Device string, e.g., cuda:0, cuda:1")
    ap.add_argument("--fail_fast", action="store_true", help="Stop on first error instead of skipping the parquet file")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process_all_parquets(
        INPUT_DIR,
        PROCESSED_DIR,
        shard=args.shard,
        num_shards=args.num_shards,
        device=args.device,
        fail_fast=args.fail_fast,
    )
