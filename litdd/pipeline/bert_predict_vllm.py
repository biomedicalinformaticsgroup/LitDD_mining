#!/usr/bin/env python3
import argparse
import gc
import os
import re
import traceback
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import load_dataset

# vLLM
from vllm import LLM

# -------------------
# Config
# -------------------

# Released LitDD screen (a BioClinical-ModernBERT-large fine-tune); needs transformers >= 4.48
# and a vLLM build with ModernBERT sequence-classification pooling.
MODEL_ID = os.environ.get("MODEL_ID", "tmy100000001/LitDD_BERT")

DEFAULT_INPUT_DIR = "data/pubmed_download/parquet_download_files"
DEFAULT_PROCESSED_DIR = "data/bert_processed"

ROW_BATCH_SIZE = 8192     # rows pulled from streaming dataset at a time (CPU-side)
PRED_BATCH_SIZE = 1024    # how many strings to send to vLLM per call (tune per GPU)
# The screen is ModernBERT: 8,192-token context. The old 512 cap was a relic of the
# BERT-large base it replaced and truncated ~1% of abstracts (observed max ~800 tokens),
# so in practice this now truncates nothing. vLLM/ModernBERT unpad, so a larger cap costs
# no throughput on short sequences.
MAX_LENGTH = 8192
PARQUET_COMPRESSION = "zstd"  # "snappy" for faster IO, larger files
SKIP_IF_EXISTS = True

# Optional CUDA perf toggles; harmless if not available.
if torch.cuda.is_available():
    try:
        # TF32 is not directly used by vLLM here, but enabling is harmless.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Not critical for vLLM, but won't harm.
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

def _token_overhead(tokenizer) -> int:
    # Estimate how many special tokens are added (e.g., [CLS], [SEP])
    sample = "x"
    with_special = tokenizer.encode(sample, add_special_tokens=True)
    without_special = tokenizer.encode(sample, add_special_tokens=False)
    return max(0, len(with_special) - len(without_special))

def _truncate_to_token_limit(tokenizer, text: str, max_tokens: int) -> str:
    # Truncate by tokens: encode without specials, slice, decode back
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > max_tokens:
        ids = ids[:max_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)

def safe_pubdate_gt_1980(x: Dict[str, Any]) -> bool:
    try:
        pd = x.get("pubdate", None)
        pd = int(pd) if pd is not None else -1
    except Exception:
        pd = -1
    # Membership, not equality. MEDLINE records multiple languages as a delimited string
    # (`eng;spa`, `por;eng`, ...), so an exact match silently discarded 134,817 English
    # records in the 2026 corpus -- measured, not estimated. These are genuinely English
    # papers and the screen can read them.
    langs = x.get("languages") or ""
    return ("eng" in langs) and (pd > 1980)


def make_tiab(x: Dict[str, Any]) -> Dict[str, Any]:
    title = x.get("title", "") or ""
    abstract = x.get("abstract", "") or ""
    x["tiab"] = f"{title} {abstract}".strip()
    return x


def argmax_index(values: List[float]) -> int:
    # Argmax over logits; no softmax needed for the predicted class.
    # Using pure Python to avoid extra tensor roundtrips.
    max_i, max_v = 0, float("-inf")
    for i, v in enumerate(values):
        if v > max_v:
            max_i, max_v = i, v
    return max_i


def predict_batch_vllm(
    llm: LLM,
    texts: List[str],
    pred_bs: int = PRED_BATCH_SIZE,
    tokenizer=None,
    text_token_limit: Optional[int] = None,
) -> List[int]:
    preds: List[int] = []
    for i in range(0, len(texts), pred_bs):
        sub = texts[i:i + pred_bs]
        if tokenizer is not None and text_token_limit is not None:
            sub = [_truncate_to_token_limit(tokenizer, t, text_token_limit) for t in sub]

        results = llm.classify(sub)
        for out in results:
            logits = getattr(out.outputs, "probs", None)
            preds.append(-1 if logits is None else argmax_index(logits))
    return preds


# -------------------
# Schema utilities 
# -------------------

def get_output_schema(parquet_path: str) -> pa.Schema:
    base = pq.read_schema(parquet_path)
    fields = list(base)
    if "tiab" not in base.names:
        fields.append(pa.field("tiab", pa.string()))
    if "bert_predict" not in base.names:
        fields.append(pa.field("bert_predict", pa.int64()))
    return pa.schema(fields)


def table_from_batch_with_schema(
    batch: Dict[str, List[Any]],
    preds: List[int],
    schema: pa.Schema
) -> pa.Table:
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


def process_one_parquet_with_tokenizer(
    parquet_path: str,
    out_dir: str,
    llm: LLM,
    tokenizer,
    text_token_limit: int,
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

                preds = predict_batch_vllm(llm, texts, tokenizer=tokenizer, text_token_limit=text_token_limit)
                table = table_from_batch_with_schema(batch, preds, out_schema)

                if writer is None:
                    writer = pq.ParquetWriter(out_path, schema=out_schema, compression=PARQUET_COMPRESSION)

                writer.write_table(table)
                total_rows += table.num_rows

                del batch, table, preds, texts
                gc.collect()
                if torch.cuda.is_available():
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



def load_vllm_engine(model_id: str, max_length: int, tp_size: int = 1,
                     dtype: str = "float32") -> LLM:
    """Load the screen under vLLM.

    dtype defaults to float32, matching the numerics the released checkpoint was trained,
    locked and evaluated in. Measured on the released checkpoint over all 2,779 ds_test rows
    (revision/vllm_modernbert_check.csv, 1xH100, vLLM 0.23.0):

        fp32  F1 0.9206  positive-rate 26.23%  FPR 3.549%  183 rows/s/GPU
        bf16  F1 0.9213  positive-rate 26.27%  FPR 3.549%  240 rows/s/GPU

    i.e. bf16 is numerically fine -- one disagreement in 2,779, 0.036pp on positive rate --
    but fp32 costs only ~24% throughput, and at that price there is no reason to deploy in
    different numerics from the published artefact. Note F1 is a weak guide here: across
    training seeds F1 moves 0.0013 while the deployment FPR proxy moves 5.00-10.37%, and over
    ~35M records 0.1pp of positive rate is ~35,000 records. Choose on positive-rate
    stability, not F1.

    Pass dtype="bfloat16" to trade that provenance tidiness for throughput.
    """
    # vLLM renamed the pooling-model selector between versions: `task="classify"` works up to
    # ~0.10.x, but 0.23.0 removed it from EngineArgs (raising TypeError) in favour of
    # `runner`/`convert`. The screen must run on the newer vLLM so a single image can also
    # serve GPT-OSS-20B for the LLM stage, so try the variants in order rather than pinning
    # to one API. Verified: 0.10.1.1 accepts task=, 0.23.0 accepts runner=/convert=.
    base = dict(
        model=model_id,
        dtype=dtype,
        max_seq_len_to_capture=max_length,
        tensor_parallel_size=tp_size,
    )
    variants = [
        ("task=classify", dict(task="classify")),
        ("runner=pooling,convert=classify", dict(runner="pooling", convert="classify")),
        ("runner=pooling", dict(runner="pooling")),
        ("auto-detect", dict()),
    ]
    last_err = None
    for name, extra in variants:
        try:
            llm = LLM(**base, **extra)
            print(f"vLLM engine constructed via {name} (dtype={dtype})")
            return llm
        except TypeError as e:
            last_err = e
            continue
    raise RuntimeError(
        f"No vLLM pooling API variant accepted by this build; last error: {last_err}"
    )


def process_all_parquets(
    input_dir: str,
    processed_dir: str,
    model_id: str,
    max_length: int = MAX_LENGTH,
    shard: int = 0,
    num_shards: int = 1,
    fail_fast: bool = False,
    tp_size: int = 1,
    dtype: str = "float32",
):
    llm = load_vllm_engine(model_id, max_length, tp_size=tp_size, dtype=dtype)

    # Get tokenizer once and compute safe text token budget
    tokenizer = llm.get_tokenizer()
    overhead = _token_overhead(tokenizer)
    text_token_limit = max(1, max_length - overhead)
    print(f"Token budget: max_length={max_length}, overhead={overhead}, text_token_limit={text_token_limit}")

    files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".parquet")])
    if not files:
        print(f"No parquet files found in {input_dir}")
        return

    files = [p for i, p in enumerate(files) if i % max(num_shards, 1) == shard]

    for path in files:
        print(f"[shard {shard}/{num_shards}] Processing: {path}")
        try:
            ok = process_one_parquet_with_tokenizer(path, processed_dir, llm, tokenizer, text_token_limit)
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

    del llm, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=MODEL_ID, help="ModernBERT classifier model id or path for vLLM")
    ap.add_argument("--shard", type=int, default=0, help="Shard index for file list")
    ap.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    ap.add_argument("--max_length", type=int, default=MAX_LENGTH, help="Max sequence length for vLLM (truncation)")
    ap.add_argument("--input_dir", default=DEFAULT_INPUT_DIR, help="Parquet shards to classify")
    ap.add_argument("--processed_dir", default=DEFAULT_PROCESSED_DIR, help="Output directory")
    ap.add_argument("--dtype", default="float32",
                    help="vLLM dtype. Default float32 -- the numerics the released "
                         "checkpoint was trained, locked and evaluated in. bfloat16 is "
                         "~1.3x faster and differs on 1/2779 test rows (0.036pp positive "
                         "rate); see load_vllm_engine() for the measurement.")
    ap.add_argument("--fail_fast", action="store_true", help="Stop on first error instead of skipping the parquet file")
    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="CUDA device(s) to use, e.g. '0' or '0,1' or 'cuda:0'. "
             "If unset, uses CUDA_VISIBLE_DEVICES or all GPUs."
    )
    return ap.parse_args()

def normalize_device_arg(device: Optional[str]) -> Optional[str]:
    if device is None:
        return None
    # Accept forms like "cuda:0", "cuda:0,1", "0", "0,1"
    dev = device.strip()
    dev = re.sub(r"^cuda:", "", dev)  # remove leading "cuda:"
    dev = dev.replace(" ", "")
    if not re.fullmatch(r"\d+(,\d+)*", dev):
        raise ValueError(f"Invalid --device value: {device}. Use e.g. '0' or '0,1' or 'cuda:0'")
    return dev

if __name__ == "__main__":
    args = parse_args()
    visible = normalize_device_arg(args.device)
    if visible is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible

    vis_env = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    tp_size = 1
    if vis_env:
        tp_size = len([v for v in vis_env.split(",") if v.strip() != ""])

    process_all_parquets(
        args.input_dir,
        args.processed_dir,
        model_id=args.model,
        max_length=args.max_length,
        shard=args.shard,
        num_shards=args.num_shards,
        fail_fast=args.fail_fast,
        tp_size=tp_size,  # pass through
        dtype=args.dtype,
    )
