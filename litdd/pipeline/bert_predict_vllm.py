#!/usr/bin/env python3
"""Screen stage: classify every eligible PubMed record with the LitDD screen under vLLM.

Reads
    ``--input_dir``     parquet shards from ``pubmed_to_parquet.py``
    ``--keep_parquet``  ``pmid_keep.parquet`` from ``dedupe_pmids.py`` (optional); when
                        given, only PMIDs listed for a shard are screened

Writes
    ``<processed_dir>/<shard>_bert_processed.parquet`` per input shard: the input columns
    plus ``tiab`` (title and abstract), ``bert_predict`` (argmax class) and ``bert_score``
    (positive-class probability). ``build_bert_positives.py`` collects the positives.

Rows are eligible when the record is in English and published after 1980
(``screen_io.safe_pubdate_gt_1980``). Each shard is written to a hidden sidecar file and
renamed into place only once the whole shard has been classified, so an interrupted run
leaves no partial output under the final name and a restart skips only complete shards.

The model is ``tmy100000001/LitDD_BERT``, a ModernBERT sequence classifier served through
vLLM's pooling runner. vLLM, torch and datasets are imported inside the functions that use
them, so the module imports on a machine without them.

Usage:
    python -m litdd.pipeline.bert_predict_vllm --input_dir data/parquet_download_files \\
        --processed_dir data/bert_processed --keep_parquet data/pmid_keep.parquet \\
        --shard 0 --num_shards 8
"""
from __future__ import annotations

import argparse
import gc
import logging
import os
import re
import sys
import traceback
from typing import TYPE_CHECKING, Any, Callable

import pyarrow.parquet as pq

from litdd.pipeline.screen_io import (
    MAX_LENGTH,
    PARQUET_COMPRESSION,
    PRED_BATCH_SIZE,
    ROW_BATCH_SIZE,
    get_output_schema,
    make_tiab,
    safe_pubdate_gt_1980,
    select_shard,
    table_from_batch_with_schema,
)

if TYPE_CHECKING:
    import pyarrow as pa
    from vllm import LLM

logger = logging.getLogger("litdd.bert_predict_vllm")

# ModernBERT sequence classifier; needs transformers >= 4.48 and a vLLM build with
# ModernBERT sequence-classification pooling.
MODEL_ID = os.environ.get("MODEL_ID", "tmy100000001/LitDD_BERT")

DEFAULT_INPUT_DIR = "data/pubmed_download/parquet_download_files"
DEFAULT_PROCESSED_DIR = "data/bert_processed"
SKIP_IF_EXISTS = True


def _cuda_available() -> bool:
    """Return True when torch is installed and sees a CUDA device."""
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def _empty_cuda_cache() -> None:
    """Release cached CUDA memory when a GPU is present."""
    if _cuda_available():
        import torch

        torch.cuda.empty_cache()


def _enable_tf32() -> None:
    """Allow TF32 matmuls when a GPU is present; vLLM ignores these flags on most paths."""
    if not _cuda_available():
        return
    import torch

    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def _token_overhead(tokenizer) -> int:
    """Number of special tokens the tokenizer adds around a text (for example [CLS], [SEP])."""
    sample = "x"
    with_special = tokenizer.encode(sample, add_special_tokens=True)
    without_special = tokenizer.encode(sample, add_special_tokens=False)
    return max(0, len(with_special) - len(without_special))


def _truncate_to_token_limit(tokenizer, text: str, max_tokens: int) -> str:
    """Encode ``text`` without special tokens, cut to ``max_tokens`` and decode back."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > max_tokens:
        ids = ids[:max_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def load_keep_pmids(keep_parquet: str) -> dict[str, set]:
    """Load ``pmid_keep.parquet`` as ``{source_shard: {pmid, ...}}``.

    The manifest written by ``dedupe_pmids.py`` names exactly one source shard per surviving
    PMID, so each input parquet consults only its own PMIDs and a record kept from a different
    shard is dropped here.
    """
    tbl = pq.read_table(keep_parquet, columns=["pmid", "source_shard"])
    by_shard: dict[str, set] = {}
    for pmid, src in zip(tbl["pmid"].to_pylist(), tbl["source_shard"].to_pylist()):
        by_shard.setdefault(src, set()).add(pmid)
    logger.info("%d PMIDs across %d source shards from %s", tbl.num_rows, len(by_shard), keep_parquet)
    return by_shard


def make_row_filter(keep_for_shard: set | None) -> Callable[[dict[str, Any]], bool]:
    """Eligibility predicate: English and post-1980, and on the keep-list when one is given."""
    if keep_for_shard is None:
        return safe_pubdate_gt_1980

    def _keep(x: dict[str, Any]) -> bool:
        return safe_pubdate_gt_1980(x) and (x.get("pmid") in keep_for_shard)

    return _keep


def argmax_index(values: list[float]) -> int:
    """Index of the largest value; ties resolve to the first."""
    max_i, max_v = 0, float("-inf")
    for i, v in enumerate(values):
        if v > max_v:
            max_i, max_v = i, v
    return max_i


def predict_batch_vllm(
    llm: LLM,
    texts: list[str],
    pred_bs: int = PRED_BATCH_SIZE,
    tokenizer=None,
    text_token_limit: int | None = None,
) -> tuple[list[int], list[float]]:
    """Classify ``texts`` in chunks of ``pred_bs``; return (labels, positive-class probabilities).

    Index 1 of the probability vector is the positive class. A result without probabilities
    yields label -1 and score NaN.
    """
    preds: list[int] = []
    scores: list[float] = []
    for i in range(0, len(texts), pred_bs):
        sub = texts[i:i + pred_bs]
        if tokenizer is not None and text_token_limit is not None:
            sub = [_truncate_to_token_limit(tokenizer, t, text_token_limit) for t in sub]

        results = llm.classify(sub)
        for out in results:
            probs = getattr(out.outputs, "probs", None)
            if probs is None:
                preds.append(-1)
                scores.append(float("nan"))
            else:
                preds.append(argmax_index(probs))
                scores.append(float(probs[1]) if len(probs) > 1 else float(probs[0]))
    return preds, scores


def _output_paths(parquet_path: str, out_dir: str) -> tuple[str, str]:
    """Return (final output path, sidecar path) for one input shard."""
    stem = os.path.splitext(os.path.basename(parquet_path))[0]
    out_path = os.path.join(out_dir, f"{stem}_bert_processed.parquet")
    tmp_path = os.path.join(out_dir, f".{stem}_bert_processed.parquet.partial")
    return out_path, tmp_path


def _remove_file(path: str, what: str) -> None:
    """Delete ``path`` if it exists, logging a warning on failure."""
    if not os.path.exists(path):
        return
    try:
        os.remove(path)
        logger.info("Removed %s: %s", what, path)
    except OSError:
        logger.warning("Could not remove %s: %s", what, path)
        traceback.print_exc()


def _open_output(parquet_path: str, out_dir: str) -> tuple[str, str] | None:
    """Prepare the output location for a shard.

    Returns ``None`` when a complete output already exists and may be skipped. Otherwise
    removes any sidecar left by an interrupted run and returns (final path, sidecar path).
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path, tmp_path = _output_paths(parquet_path, out_dir)
    if SKIP_IF_EXISTS and os.path.exists(out_path):
        logger.info("Skipping (already exists): %s", out_path)
        return None
    _remove_file(tmp_path, "incomplete shard from an earlier run")
    return out_path, tmp_path


def _open_dataset(parquet_path: str, keep_for_shard: set | None):
    """Stream ``parquet_path`` as batches of eligible rows with a ``tiab`` field."""
    from datasets import load_dataset

    ds = load_dataset("parquet", data_files=parquet_path, split="train", streaming=True)
    ds = ds.filter(make_row_filter(keep_for_shard))
    ds = ds.map(make_tiab)
    return ds.batch(ROW_BATCH_SIZE)


def _classify_batches(
    ds,
    tmp_path: str,
    out_schema: pa.Schema,
    llm: LLM,
    tokenizer,
    text_token_limit: int,
    parquet_path: str,
) -> tuple[int, bool]:
    """Classify every batch of ``ds`` and append it to the sidecar parquet.

    Returns (rows written, failed). On the first failing batch the loop stops and ``failed``
    is True; the writer is closed in every case.
    """
    writer = None
    total_rows = 0
    failed = False
    try:
        for batch in ds:
            try:
                texts = batch.get("tiab", [])
                if not texts:
                    continue
                preds, scores = predict_batch_vllm(
                    llm, texts, tokenizer=tokenizer, text_token_limit=text_token_limit)
                table = table_from_batch_with_schema(batch, out_schema, preds, scores)
                if writer is None:
                    writer = pq.ParquetWriter(tmp_path, schema=out_schema,
                                              compression=PARQUET_COMPRESSION)
                writer.write_table(table)
                total_rows += table.num_rows

                del batch, table, preds, scores, texts
                gc.collect()
                _empty_cuda_cache()
            except Exception:
                failed = True
                logger.error("Failed processing a batch in: %s", parquet_path)
                traceback.print_exc()
                break
    except Exception:
        failed = True
        logger.error("Iteration over dataset failed for: %s", parquet_path)
        traceback.print_exc()
    finally:
        try:
            if writer is not None:
                writer.close()
        except Exception:
            logger.warning("Failed to close writer for: %s", tmp_path)
            traceback.print_exc()
    return total_rows, failed


def _publish(tmp_path: str, out_path: str, total_rows: int, failed: bool, parquet_path: str) -> bool:
    """Rename the sidecar to the final name, or remove it when the shard failed or was empty.

    ``os.replace`` is atomic within a filesystem, so the final name appears only for a
    complete shard. Returns False when the shard failed or the rename failed.
    """
    if failed:
        _remove_file(tmp_path, "partial output")
        _empty_cuda_cache()
        return False
    if total_rows == 0:
        logger.info("No eligible rows in %s; no output written.", parquet_path)
        _remove_file(tmp_path, "empty output")
        return True
    try:
        os.replace(tmp_path, out_path)
    except OSError:
        logger.error("Failed to publish %s -> %s", tmp_path, out_path)
        traceback.print_exc()
        return False
    logger.info("Wrote %d rows to %s", total_rows, out_path)
    return True


def process_one_parquet_with_tokenizer(
    parquet_path: str,
    out_dir: str,
    llm: LLM,
    tokenizer,
    text_token_limit: int,
    keep_for_shard: set | None = None,
) -> bool:
    """Screen one input shard into ``out_dir``. Returns True on success or skip."""
    paths = _open_output(parquet_path, out_dir)
    if paths is None:
        return True
    out_path, tmp_path = paths

    try:
        ds = _open_dataset(parquet_path, keep_for_shard)
    except Exception:
        logger.error("Failed to open or prepare dataset for: %s", parquet_path)
        traceback.print_exc()
        return False

    try:
        out_schema = get_output_schema(parquet_path)
    except Exception:
        logger.error("Failed to read schema from: %s", parquet_path)
        traceback.print_exc()
        return False

    total_rows, failed = _classify_batches(
        ds, tmp_path, out_schema, llm, tokenizer, text_token_limit, parquet_path)
    return _publish(tmp_path, out_path, total_rows, failed, parquet_path)


def load_vllm_engine(model_id: str, max_length: int, tp_size: int = 1,
                     dtype: str = "float32") -> LLM:
    """Construct the vLLM engine for the screen model.

    ``dtype`` defaults to float32, the numerics the released checkpoint was trained and
    evaluated in; ``"bfloat16"`` is faster. vLLM builds differ in how a pooling model is
    selected, so the keyword variants are tried in order and the first accepted one is used.
    """
    from vllm import LLM

    _enable_tf32()
    base = dict(
        model=model_id,
        dtype=dtype,
        max_model_len=max_length,
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
            logger.info("vLLM engine constructed via %s (dtype=%s)", name, dtype)
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
    keep_parquet: str | None = None,
) -> None:
    """Screen the shard's share of the parquet files in ``input_dir`` into ``processed_dir``."""
    keep_by_shard = load_keep_pmids(keep_parquet) if keep_parquet else None
    llm = load_vllm_engine(model_id, max_length, tp_size=tp_size, dtype=dtype)

    # Reserve room for the special tokens the tokenizer adds.
    tokenizer = llm.get_tokenizer()
    overhead = _token_overhead(tokenizer)
    text_token_limit = max(1, max_length - overhead)
    logger.info("Token budget: max_length=%d, overhead=%d, text_token_limit=%d",
                max_length, overhead, text_token_limit)

    files = sorted(os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".parquet"))
    if not files:
        logger.warning("No parquet files found in %s", input_dir)
        return
    files = select_shard(files, shard, num_shards)

    for path in files:
        logger.info("[shard %d/%d] Processing: %s", shard, num_shards, path)
        keep_for_shard = None
        if keep_by_shard is not None:
            keep_for_shard = keep_by_shard.get(os.path.basename(path), set())
            if not keep_for_shard:
                logger.info("No keep-listed PMIDs for %s; skipping.", os.path.basename(path))
                continue
        try:
            ok = process_one_parquet_with_tokenizer(path, processed_dir, llm, tokenizer,
                                                    text_token_limit, keep_for_shard)
            if not ok:
                if fail_fast:
                    raise RuntimeError(f"Stopping due to error on file: {path}")
                logger.warning("Skipping file due to error: %s", path)
                continue
        except Exception:
            logger.error("Unhandled exception while processing: %s", path)
            traceback.print_exc()
            if fail_fast:
                raise

    del llm, tokenizer
    gc.collect()
    _empty_cuda_cache()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line interface of the screen stage."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", type=str, default=MODEL_ID,
                    help="Screen classifier model id or path for vLLM")
    ap.add_argument("--shard", type=int, default=0, help="Shard index for the file list")
    ap.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    ap.add_argument("--max_length", type=int, default=MAX_LENGTH,
                    help="Max sequence length for vLLM (longer texts are truncated)")
    ap.add_argument("--input_dir", default=DEFAULT_INPUT_DIR, help="Parquet shards to classify")
    ap.add_argument("--processed_dir", default=DEFAULT_PROCESSED_DIR, help="Output directory")
    ap.add_argument("--dtype", default="float32",
                    help="vLLM dtype: float32 (default, the numerics of the released "
                         "checkpoint) or bfloat16 (faster).")
    ap.add_argument("--keep_parquet", type=str, default=None,
                    help="pmid_keep.parquet from dedupe_pmids.py. Restricts the screen to the "
                         "de-duplicated corpus with withdrawn and retracted records removed.")
    ap.add_argument("--fail_fast", action="store_true",
                    help="Stop on the first error instead of skipping the parquet file")
    ap.add_argument("--device", type=str, default=None,
                    help="CUDA device(s) to use, e.g. '0' or '0,1' or 'cuda:0'. "
                         "If unset, uses CUDA_VISIBLE_DEVICES or all GPUs.")
    return ap.parse_args(argv)


def normalize_device_arg(device: str | None) -> str | None:
    """Reduce ``'cuda:0,1'``, ``'0, 1'`` or ``'0'`` to a comma-separated index list."""
    if device is None:
        return None
    dev = device.strip()
    dev = re.sub(r"^cuda:", "", dev)
    dev = dev.replace(" ", "")
    if not re.fullmatch(r"\d+(,\d+)*", dev):
        raise ValueError(f"Invalid --device value: {device}. Use e.g. '0' or '0,1' or 'cuda:0'")
    return dev


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

    # Tensor parallelism spans every visible device.
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
        tp_size=tp_size,
        dtype=args.dtype,
        keep_parquet=args.keep_parquet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
