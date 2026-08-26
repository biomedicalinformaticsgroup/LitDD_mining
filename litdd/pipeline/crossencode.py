#!/usr/bin/env python3
"""Cross-encoder ranking of (TIAB, G2P_LGMDE) candidate pairs.

Cross-encoder similarity scoring
--------------------------------
For every BERT-positive PubMed abstract (``tiab``) and every G2P record we
assemble a single ``(TIAB, G2P_LGMDE)`` text pair, where ``G2P_LGMDE`` is the
record concatenated as
``g2p_id - gene symbol - gene mim - hgnc id - previous gene symbols -
disease name - disease mim - disease MONDO - allelic requirement -
cross cutting modifier - confidence - inferred variant consequence -
variant types - molecular mechanism - molecular mechanism categorisation``
(see ``build_g2p_lgmde_list`` below).

The cross-encoder is a single transformer (a fine-tuned ``ncbi/MedCPT-Cross-
Encoder``) that ingests the joined pair via ``[CLS] tiab [SEP] g2p_lgmde
[SEP]`` and a 1-logit classification head. ``model.predict`` applies a
sigmoid on top of that logit and returns a relevance score in ``[0, 1]`` —
higher means more semantically related. We retain the top-5 scoring G2P
records per TIAB; those candidates are passed to the LLM mapping stage and
the score itself is later thresholded in ``final_data_clean.py``
(default cutoff 0.9).

Computationally we score the ``N × M`` pair grid in row-blocks: G2P records
are broken into ``g_block_size`` chunks, each (text, candidate) batch is
predicted in ``pair_batch_size`` micro-batches, and per-text min-heaps keep
only the top-5 across all G2P chunks.
"""
from __future__ import annotations

import argparse
import gc
import heapq
import os
import traceback
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl

# NOTE: torch and sentence_transformers are imported lazily inside the functions that
# need them, so the pure ranking helpers (top-k heap update, pair building, G2P LGMDE
# string construction) can be imported and unit-tested without the GPU stack installed.
# Type-only imports keep the forward-ref annotations resolvable without a runtime import.
if TYPE_CHECKING:
    import torch  # noqa: F401
    from sentence_transformers import CrossEncoder  # noqa: F401


# Fine-tuned ncbi/MedCPT-Cross-Encoder. Override with --model_path / $LITDD_CROSSENCODER.
DEFAULT_MODEL_PATH = os.environ.get("LITDD_CROSSENCODER", "tmy100000001/LitDD_crossencoder")
DEFAULT_INPUT_PARQUET = "pubmed_bert_positive.parquet"
DEFAULT_OUT_DIR = "crossencoded_shards"
PARQUET_COMPRESSION = "zstd"
SKIP_IF_EXISTS = True

def get_device(device_str: Optional[str] = None) -> str:
    import torch
    if device_str:
        return device_str
    return "cuda" if torch.cuda.is_available() else "cpu"


def pick_torch_dtype(dtype_str: str = "auto") -> Optional["torch.dtype"]:
    import torch
    if dtype_str == "fp32":
        return torch.float32
    if dtype_str == "fp16":
        return torch.float16
    if dtype_str == "bf16":
        return torch.bfloat16
    
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return None  


def load_crossencoder(
    model_path: str,
    device_str: Optional[str] = None,
    dtype_str: str = "auto",
) -> Tuple["CrossEncoder", str]:
    import torch
    from sentence_transformers import CrossEncoder

    device = get_device(device_str)
    dtype = pick_torch_dtype(dtype_str)
    model_kwargs = {}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    model = CrossEncoder(
        model_path,
        device=device,
        model_kwargs=model_kwargs,
    )

    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    return model, device


def build_g2p_lgmde_list(g2p_csv_path: str) -> List[str]:
    """Unique LGMDE candidate threads for the cross-encoder to rank against.

    Delegates to ``litdd.threads``, which is shared with the training path. These used to be
    two independent implementations that disagreed -- inference looked for a column named
    "inferred variant consequence" that exists in no G2P export, so that field was silently
    blank in every candidate served to a model fine-tuned on strings where it was populated.
    See ``litdd/threads.py`` for the full account.
    """
    from litdd.threads import build_lgmde_list

    return build_lgmde_list(g2p_csv_path)


def load_shard_df(input_parquet: str, shard: int, num_shards: int) -> pl.DataFrame:
    # read only rows for this shard, preserving original global order via row_nr
    df_shard = (
        pl.scan_parquet(input_parquet)
        .with_row_index(name="row_nr")  # instead of deprecated with_row_count
        .filter((pl.col("row_nr") % pl.lit(num_shards)) == pl.lit(shard))
        .sort("row_nr")
        .collect(engine="streaming")
    )
    return df_shard


def make_pairs_for_block(texts: List[str], g_block: List[str]) -> List[Tuple[str, str]]:
    # for each text, iterate all g2p in block -> enables reshape(C, L)
    pairs: List[Tuple[str, str]] = []
    for t in texts:
        pairs.extend((t, g) for g in g_block)
    return pairs


def update_topk_heaps_from_block(
    heaps: List[List[Tuple[float, str]]],  # min-heaps of (score, label)
    scores_block: np.ndarray,              # shape (C, L_block)
    g_block: List[str],
    k: int,
) -> None:
    C, L = scores_block.shape
    g_block_arr = np.array(g_block, dtype=object)
    tk = min(k, L)
    for i in range(C):
        row = scores_block[i]
        if tk < L:
            idxs = np.argpartition(row, -tk)[-tk:]
        else:
            idxs = np.arange(L)
        heap = heaps[i]
        for j in idxs:
            s = float(row[j])
            label = g_block_arr[j]
            if len(heap) < k:
                heapq.heappush(heap, (s, label))
            else:
                if s > heap[0][0]:
                    heapq.heapreplace(heap, (s, label))


def crossencode_topk_for_chunk(
    model: "CrossEncoder",
    chunk_texts: List[str],
    g2p_list: List[str],
    top_k: int = 5,
    pair_batch_size: int = 256,
    g_block_size: int = 2048,
) -> List[List[Dict[str, Any]]]:
    import torch

    C = len(chunk_texts)
    if C == 0:
        return []

    # One min-heap per text to track best K
    heaps: List[List[Tuple[float, str]]] = [[] for _ in range(C)]

    # Iterate the G2P space in blocks to keep memory under control
    for start in range(0, len(g2p_list), g_block_size):
        end = min(start + g_block_size, len(g2p_list))
        g_block = g2p_list[start:end]

        # Build pairs in order [ (t0,g0..gL-1), (t1,g0..), ... ] 
        pairs = make_pairs_for_block(chunk_texts, g_block)

        # Predict in batches via CrossEncoder; returns (len(pairs),)
        scores_flat = model.predict(pairs, batch_size=pair_batch_size)
        scores_block = np.asarray(scores_flat).reshape(C, len(g_block))

        # Merge top-k from this block
        update_topk_heaps_from_block(heaps, scores_block, g_block, top_k)

        del pairs, scores_flat, scores_block
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Convert heaps to sorted desc lists of dicts: [{"label": ..., "score": ...}, ...]
    topk_lists: List[List[Dict[str, Any]]] = []
    for heap in heaps:
        sorted_desc = sorted(heap, key=lambda x: x[0], reverse=True)
        topk_lists.append(
            [{"label": label, "score": float(score)} for (score, label) in sorted_desc]
        )
    return topk_lists


def process_shard_candidates(
    candidates_parquet: str,
    g2p_csv: str,
    out_dir: str,
    model_path: str = DEFAULT_MODEL_PATH,
    device: Optional[str] = None,
    dtype: str = "auto",
    pair_batch_size: int = 256,
    top_k: Optional[int] = None,
    shard: int = 0,
    num_shards: int = 1,
    skip_if_exists: bool = SKIP_IF_EXISTS,
    compression: str = PARQUET_COMPRESSION,
) -> bool:
    """Score only the candidates ``gene_candidates.py`` selected for each abstract.

    The full-grid path scores every abstract against all ~2,861 G2P entries and keeps a top-k
    by min-heap. Once the gene gate has run, each abstract carries a handful of candidates
    (mean ~3.5, median 2), so the heap machinery is unnecessary: we build the explicit pair
    list and score it directly, which is ~3,600x less work.

    ``top_k=None`` keeps every candidate -- the data-driven setting (R3.5). Pass an integer to
    reproduce a fixed-k cascade for comparison.

    The output column keeps the name ``top5_cross`` and the ``list<struct<label, score>>``
    layout so ``llm_map.py`` and ``final_data_clean.py`` are unchanged, and so existing
    deployed parquets remain readable by the same code.
    """
    import torch  # noqa: F401  (kept for parity with the full-grid path's import contract)

    from litdd.threads import build_lgmde_map

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(candidates_parquet))[0]
    out_path = os.path.join(out_dir, f"{stem}_crossencoded_shard{shard}-of-{num_shards}.parquet")
    if skip_if_exists and os.path.exists(out_path):
        print(f"Skipping (already exists): {out_path}")
        return True

    thread_by_id = build_lgmde_map(g2p_csv)
    print(f"[INFO] G2P threads: {len(thread_by_id)}")

    df = load_shard_df(candidates_parquet, shard, num_shards)
    if "candidate_g2p_ids" not in df.columns:
        raise SystemExit(
            f"{candidates_parquet} has no 'candidate_g2p_ids' column -- run "
            "litdd/pipeline/gene_candidates.py first, or use the full-grid path without "
            "--candidates_parquet."
        )
    print(f"[INFO] Shard rows (N_shard): {df.height}")

    texts = df["tiab"].to_list()
    cand_ids = df["candidate_g2p_ids"].to_list()

    # Flatten to one pair list so the model sees full batches regardless of how uneven the
    # per-row candidate counts are.
    pairs: List[Tuple[str, str]] = []
    owner: List[int] = []
    labels: List[str] = []
    for i, (text, ids) in enumerate(zip(texts, cand_ids)):
        for gid in (ids or []):
            thread = thread_by_id.get(gid)
            if thread is None:
                continue
            pairs.append((text or "", thread))
            owner.append(i)
            labels.append(thread)
    print(f"[INFO] pairs to score: {len(pairs):,}")

    model, _ = load_crossencoder(model_path, device=device, dtype=dtype)
    scores: List[float] = []
    for i in range(0, len(pairs), pair_batch_size):
        out = model.predict(pairs[i:i + pair_batch_size])
        scores.extend(float(x) for x in np.asarray(out).reshape(-1))

    per_row: List[List[dict]] = [[] for _ in range(df.height)]
    for idx, score in enumerate(scores):
        per_row[owner[idx]].append({"label": labels[idx], "score": score})
    for row in per_row:
        row.sort(key=lambda d: d["score"], reverse=True)
    if top_k is not None:
        per_row = [row[:top_k] for row in per_row]

    df = df.with_columns(pl.Series("top5_cross", per_row))
    df.write_parquet(out_path, compression=compression)
    print(f"[INFO] wrote {out_path}")
    return True


def process_shard(
    input_parquet: str,
    g2p_csv: str,
    out_dir: str,
    model_path: str = DEFAULT_MODEL_PATH,
    device: Optional[str] = None,
    dtype: str = "auto",
    chunk_size: int = 200,
    pair_batch_size: int = 256,
    g_block_size: int = 2048,
    top_k: int = 5,
    shard: int = 0,
    num_shards: int = 1,
    skip_if_exists: bool = SKIP_IF_EXISTS,
    compression: str = PARQUET_COMPRESSION,
) -> bool:
    import torch

    os.makedirs(out_dir, exist_ok=True)

    base = os.path.basename(input_parquet)
    stem = os.path.splitext(base)[0]
    out_path = os.path.join(out_dir, f"{stem}_crossencoded_shard{shard}-of-{num_shards}.parquet")

    if skip_if_exists and os.path.exists(out_path):
        print(f"Skipping (already exists): {out_path}")
        return True

    # Load G2P list
    try:
        print(f"[INFO] Loading G2P list from: {g2p_csv}")
        unique_lgmde = build_g2p_lgmde_list(g2p_csv)
    except Exception:
        print(f"[ERROR] Failed to load/prepare G2P CSV: {g2p_csv}")
        traceback.print_exc()
        return False

    M = len(unique_lgmde)
    print(f"[INFO] G2P unique entries (M): {M}")

    # Load shard of the input parquet
    try:
        print(f"[INFO] Loading shard {shard}/{num_shards} from: {input_parquet}")
        df_shard = load_shard_df(input_parquet, shard, num_shards)
    except Exception:
        print(f"[ERROR] Failed to load shard {shard}/{num_shards} from {input_parquet}")
        traceback.print_exc()
        return False

    if df_shard.height == 0:
        print(f"[INFO] No rows for shard {shard}/{num_shards}; nothing to do.")
        return True

    tiab_list = df_shard.get_column("tiab").to_list()
    N = len(tiab_list)
    print(f"[INFO] Shard rows (N_shard): {N}")

    # Load model
    try:
        print(f"[INFO] Loading CrossEncoder from: {model_path}")
        model, device_str = load_crossencoder(model_path, device, dtype)
        print(f"[INFO] Model loaded on device: {device_str}")
    except Exception:
        print(f"[ERROR] Failed to load CrossEncoder: {model_path}")
        traceback.print_exc()
        return False

    # Compute top-k per row in chunks
    all_topk: List[Optional[List[Dict[str, Any]]]] = [None] * N

    try:
        total_chunks = (N + chunk_size - 1) // chunk_size
        for ci, chunk_start in enumerate(range(0, N, chunk_size), start=1):
            chunk_end = min(chunk_start + chunk_size, N)
            print(f"[INFO] Processing chunk {ci}/{total_chunks}: rows {chunk_start}:{chunk_end}")
            chunk_texts = tiab_list[chunk_start:chunk_end]

            topk_chunk = crossencode_topk_for_chunk(
                model=model,
                chunk_texts=chunk_texts,
                g2p_list=unique_lgmde,
                top_k=top_k,
                pair_batch_size=pair_batch_size,
                g_block_size=g_block_size,
            )

            all_topk[chunk_start:chunk_end] = topk_chunk
            print(f"[INFO] Finished chunk {ci}/{total_chunks}")

            del chunk_texts, topk_chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    except Exception:
        print(f"[ERROR] Failure during crossencoding for shard {shard}")
        traceback.print_exc()
        return False

    # Attach result column and write shard parquet
    try:
        # Define explicit dtype for List(Struct(label: Utf8, score: Float64))
        topk_dtype = pl.List(
            pl.Struct([
                pl.Field("label", pl.Utf8),
                pl.Field("score", pl.Float64),
            ])
        )
        print(f"[INFO] Writing output to: {out_path}")
        df_out = df_shard.with_columns(
            pl.Series(name="top5_cross", values=all_topk, dtype=topk_dtype)
        )
        df_out.write_parquet(out_path, compression=compression)
        print(f"[INFO] Wrote {df_out.height} rows to {out_path}")
    except Exception:
        print(f"[ERROR] Failed to write output parquet: {out_path}")
        traceback.print_exc()
        return False
    finally:
        del df_shard
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return True




def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_parquet", type=str, default=DEFAULT_INPUT_PARQUET, help="Path to input parquet with a 'tiab' column")
    ap.add_argument("--g2p_csv", type=str, required=True, help="Path to the G2P CSV")
    ap.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR, help="Output directory for shard parquets")
    ap.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="CrossEncoder model path")
    ap.add_argument("--device", type=str, default=None, help="Device string, e.g., cuda:0, cuda:1")
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp32", "fp16", "bf16"], help="Model dtype")
    ap.add_argument("--chunk_size", type=int, default=200, help="Number of tiab rows to process per chunk")
    ap.add_argument("--pair_batch_size", type=int, default=256, help="CrossEncoder prediction batch size")
    ap.add_argument("--g_block_size", type=int, default=2048, help="How many G2P entries to score at once")
    ap.add_argument("--candidates_parquet", type=str, default=None,
                    help="Output of gene_candidates.py. Scores ONLY each row's "
                         "candidate_g2p_ids instead of the full G2P panel (~3,600x less "
                         "work). Mutually exclusive with the full-grid --input_parquet path.")
    ap.add_argument("--top_k", type=int, default=None,
                    help="Top-K G2P entries to keep per text. Default: 5 for the full-grid "
                         "path (which must cut something out of 2,861), and NO cap for "
                         "--candidates_parquet, where the gene gate already made the count "
                         "data-driven (R3.5). Pass an integer to force a fixed k in either.")
    ap.add_argument("--shard", type=int, default=0, help="Shard index for row-wise sharding")
    ap.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    ap.add_argument("--skip_if_exists", action="store_true", help="Skip if shard output already exists")
    ap.add_argument("--no_skip_if_exists", dest="skip_if_exists", action="store_false")
    ap.set_defaults(skip_if_exists=SKIP_IF_EXISTS)
    ap.add_argument("--compression", type=str, default=PARQUET_COMPRESSION, help="Parquet compression (e.g., zstd, snappy)")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.candidates_parquet:
        # Data-driven k unless the caller explicitly asked for a fixed cut.
        top_k = None if args.top_k in (None, -1, 0) else args.top_k
        ok = process_shard_candidates(
            candidates_parquet=args.candidates_parquet,
            g2p_csv=args.g2p_csv,
            out_dir=args.out_dir,
            model_path=args.model_path,
            device=args.device,
            dtype=args.dtype,
            pair_batch_size=args.pair_batch_size,
            top_k=top_k,
            shard=args.shard,
            num_shards=args.num_shards,
            skip_if_exists=args.skip_if_exists,
            compression=args.compression,
        )
        raise SystemExit(0 if ok else 1)

    ok = process_shard(
        input_parquet=args.input_parquet,
        g2p_csv=args.g2p_csv,
        out_dir=args.out_dir,
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        chunk_size=args.chunk_size,
        pair_batch_size=args.pair_batch_size,
        g_block_size=args.g_block_size,
        top_k=5 if args.top_k is None else args.top_k,
        shard=args.shard,
        num_shards=args.num_shards,
        skip_if_exists=args.skip_if_exists,
        compression=args.compression,
    )
    if not ok:
        raise SystemExit(1)
