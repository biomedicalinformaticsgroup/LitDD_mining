"""Adjudication stage: map each screened TIAB to G2P entries with an LLM under vLLM.

Input: the shard parquets written by ``build_llm_shards.py`` (columns ``pmid``, ``tiab``,
``candidates``, a list of G2P ids from the gene gate), the contextualised-thread JSON
(``--context_json``), the all-panel G2P export (``--panel_siblings_csv``) and the released
panel export (``--final_panel_csv``).

Output: ``{shard}[_w{shard_index}]__llm.parquet`` with the input columns followed by
``candidate_text`` (the candidate blocks the model saw), ``llm_prompt``,
``generated_text``, ``llm_answer_raw``, ``llm_dis_map`` (``G2Pxxxxx``, ``G2Pa;G2Pb`` or
``NO MATCH``, restricted to the released panel), ``llm_dis_map_all_panels``,
``answer_format_valid``, ``answer_uncertain``, ``answer_ids_in_candidates``,
``finish_reason``, ``prompt_tokens`` and ``gen_tokens``; and
``{shard}[_w{shard_index}]__llm.run_meta.json`` with the settings, library versions and
throughput of the run. ``final_data_clean.py`` reads ``pmid``, ``llm_dis_map`` and
``candidates`` from the parquet.

Each row's prompt is built by ``llm_prompt`` (same-gene entries from every panel added,
ids swapped for their context blocks, panel labelled, rendered into
``prompts/decision_rubric.txt``) and sent through the model's chat template. The answer
is parsed by ``llm_answer``. Generation runs once per ``--save_every`` window and is
resumable per shard from the partially written parquet. Rows can be striped across
workers with ``--shard_index``/``--num_shards``.

The released configuration is GPT-OSS-20B at temperature 0 with medium reasoning effort;
the full command is in ``supplementary/RUN_FINAL_PIPELINE.md``.

torch and vllm are imported inside ``run_llm_over_shards`` so that the module imports and
its helpers unit-test without the GPU stack.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import glob
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from litdd.pipeline.llm_answer import NO_MATCH, extract_last_answer, parse_answer, restrict_to_panel
from litdd.pipeline.llm_prompt import (
    DEFAULT_PROMPT_FILE,
    add_panel_siblings,
    build_llm_prompt,
    candidate_ids,
    candidate_list,
    contextualise,
    drop_context_fields,
    label_panel,
    load_context_threads,
    load_panel_siblings,
)

logger = logging.getLogger(__name__)

SKIPPED_TEXT = "[skipped: no candidate passed the score gate]"
SKIPPED_TOO_LONG_TEXT = "[skipped: prompt exceeds the model context even undecorated]"

# Columns written next to the input columns, candidate_text, llm_prompt and generated_text.
EXTRA_COLS = ["llm_answer_raw", "llm_dis_map", "llm_dis_map_all_panels",
              "answer_format_valid", "answer_uncertain",
              "answer_ids_in_candidates", "finish_reason", "prompt_tokens", "gen_tokens"]


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
@dataclass
class LlmMapConfig:
    """Settings of one adjudication run. Field names equal the CLI flag names."""

    shards_dir: str
    llm_model: str
    out_dir: str | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 8192
    max_model_len: int = 16384
    gpu_memory_utilization: float = 0.90
    dtype: str = "auto"
    seed: int = 0
    reasoning_effort: str | None = "medium"
    prompt_file: str = DEFAULT_PROMPT_FILE
    context_json: str = ""
    context_drop_fields: str | None = None
    panel_siblings_csv: str | None = None
    final_panel_csv: str | None = None
    shard_index: int | None = None
    num_shards: int | None = None
    save_every: int = 1000
    tensor_parallel_size: int | None = None
    max_num_seqs: int = 512
    limit: int | None = None


@dataclass
class Resources:
    """Lookup tables loaded once per run."""

    context: dict[str, str]
    context_missing: dict[str, int]
    siblings: dict[str, dict] | None
    final_ids: set[str] | None


@dataclass
class ShardPrep:
    """Per-row derived values of one shard, aligned with the shard frame's index."""

    n_candidates: pd.Series
    skipped: pd.Series
    allowed: pd.Series


@dataclass
class ShardState:
    """Generated text and parsed columns of one shard, filled in as generation proceeds."""

    generated_texts: list[str]
    extras: dict[str, list[Any]]


# --------------------------------------------------------------------------------------
# Sharding
# --------------------------------------------------------------------------------------
def row_slice_for_worker(n_rows: int, shard_index: int | None, num_shards: int | None) -> list[int]:
    """Row indices owned by one worker, striped so every worker gets rows from every file.

    Striping rather than contiguous blocks keeps the workers balanced when prompt length
    varies along a file, and adding a worker does not reshuffle the others' spans.
    """
    if shard_index is None or num_shards is None or num_shards <= 1:
        return list(range(n_rows))
    return list(range(shard_index, n_rows, num_shards))


# --------------------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------------------
def _git_sha() -> str | None:
    """The commit of this checkout, or None when git is unavailable."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        # safe.directory lets git read a checkout owned by another user (containers run as root).
        return subprocess.run(["git", "-c", "safe.directory=*", "-C", here, "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _versions() -> dict[str, str | None]:
    """Python, vllm, torch and transformers versions (None when a package is absent)."""
    v: dict[str, str | None] = {"python": sys.version.split()[0]}
    for mod in ("vllm", "torch", "transformers"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            v[mod] = None
    return v


# --------------------------------------------------------------------------------------
# Driver steps
# --------------------------------------------------------------------------------------
def _load_resources(cfg: LlmMapConfig) -> Resources:
    """Load the context threads (with dropped fields), the panel index and the released ids."""
    context_missing: dict[str, int] = {}
    if not cfg.context_json:
        raise ValueError("--context_json is required")
    context = load_context_threads(cfg.context_json)
    logger.info("Loaded %d contextualised threads from %s", len(context), cfg.context_json)
    if cfg.context_drop_fields:
        context = drop_context_fields(context, cfg.context_drop_fields)
        logger.info("Dropped fields from contextualised threads: %s", cfg.context_drop_fields)
    siblings = load_panel_siblings(cfg.panel_siblings_csv) if cfg.panel_siblings_csv else None
    final_ids = (set(pd.read_csv(cfg.final_panel_csv, dtype=str)["g2p id"].str.strip())
                 if cfg.final_panel_csv else None)
    if siblings is not None:
        logger.info("Adding same-gene entries from every panel of %s; answers restricted to %s entries of %s",
                    cfg.panel_siblings_csv, len(final_ids) if final_ids else "all", cfg.final_panel_csv)
    return Resources(context=context, context_missing=context_missing, siblings=siblings, final_ids=final_ids)


def _prepare_shard(df: pd.DataFrame, cfg: LlmMapConfig, res: Resources) -> ShardPrep:
    """Add ``candidate_text`` and ``llm_prompt`` to the shard frame in place.

    Steps: normalise the ``candidates`` cell, append same-gene entries from other panels,
    swap ids for context blocks, label each block with its panel, render the prompt.
    Rows whose candidate list is empty get ``llm_prompt`` None and are marked skipped.
    """
    n_candidates = df["candidates"].apply(lambda x: len(candidate_list(x)))
    flat = df["candidates"].apply(candidate_list)
    if res.siblings is not None:
        n_before = int(flat.apply(len).sum())
        flat = flat.apply(lambda labs: add_panel_siblings(labs, res.siblings))
        logger.info("%d same-gene entries from other panels added to %d rows",
                    int(flat.apply(len).sum()) - n_before, len(df))
    # Rows with no candidate never reach the model: they are recorded as NO MATCH with
    # finish_reason="skipped".
    skipped = flat.apply(len) == 0
    if skipped.any():
        logger.warning("%d rows have no candidates at all (-> NO MATCH)", int(skipped.sum()))
    df["candidate_text"] = flat.apply(
        lambda labs: contextualise(labs, res.context, res.context_missing) if res.context else labs)
    if res.siblings is not None:
        df["candidate_text"] = df["candidate_text"].apply(
            lambda labs: [label_panel(lab, res.siblings) for lab in labs])
    df["llm_prompt"] = [
        build_llm_prompt(t, labs, template_path=cfg.prompt_file) if labs else None
        for t, labs in zip(df.get("tiab", pd.Series([""] * len(df))), df["candidate_text"])
    ]
    allowed = flat.apply(candidate_ids)
    return ShardPrep(n_candidates=n_candidates, skipped=skipped, allowed=allowed)


def _fit_context_budget(df: pd.DataFrame, cfg: LlmMapConfig, tokenizer: Any) -> set[int]:
    """Indices of rows whose prompt does not fit the model context.

    A prompt longer than the model context makes vLLM abort the whole batch, so such rows
    are skipped explicitly (finish_reason="too_long") instead. The budget leaves room for
    the generation: ``max_model_len`` minus the larger of 1024 and a quarter of
    ``max_tokens``. Prompts shorter than two characters per budget token are not tokenised.
    """
    budget = cfg.max_model_len - max(1024, cfg.max_tokens // 4)
    too_long_rows: set[int] = set()
    for i, prompt in enumerate(df["llm_prompt"].tolist()):
        if prompt is None or len(prompt) < budget * 2:
            continue
        if len(tokenizer.encode(prompt)) <= budget:
            continue
        too_long_rows.add(i)
    if too_long_rows:
        logger.info("over-budget prompts: %d skipped as too long", len(too_long_rows))
    return too_long_rows


def _load_resume_state(out_parquet: str, n_rows: int) -> ShardState:
    """Fresh per-row state, or the state read back from a partially written output parquet.

    An existing output is reused only when its row count matches; an unreadable file
    starts the shard afresh.
    """
    generated_texts = [""] * n_rows
    extras: dict[str, list[Any]] = {c: [None] * n_rows for c in EXTRA_COLS}
    if os.path.exists(out_parquet):
        try:
            prev = pd.read_parquet(out_parquet)
            if len(prev) == n_rows:
                generated_texts = ["" if pd.isna(t) else str(t)
                                   for t in prev["generated_text"].tolist()]
                for c in EXTRA_COLS:
                    if c in prev.columns:
                        extras[c] = prev[c].tolist()
                done = sum(1 for t in generated_texts if t)
                logger.info("%d/%d rows already generated in %s", done, n_rows, out_parquet)
            else:
                logger.info("row-count mismatch (%d != %d); starting fresh", len(prev), n_rows)
        except Exception as e:  # noqa: BLE001 - a corrupt checkpoint must not block the run
            logger.warning("could not read %s (%s); starting fresh", out_parquet, e)
    return ShardState(generated_texts=generated_texts, extras=extras)


def _save_progress(df: pd.DataFrame, out_parquet: str, state: ShardState) -> None:
    """Write the shard frame with the current generated text and parsed columns."""
    df["generated_text"] = state.generated_texts
    for c in EXTRA_COLS:
        df[c] = state.extras[c]
    df.to_parquet(out_parquet, index=False)
    logger.info("Saved current progress to %s", out_parquet)


def _mark_skipped_rows(state: ShardState, skipped: pd.Series, too_long_rows: set[int]) -> None:
    """Fill the rows that are not sent to the model: too long, or without candidates."""
    for i in too_long_rows:
        if state.generated_texts[i]:
            continue
        state.generated_texts[i] = SKIPPED_TOO_LONG_TEXT
        state.extras["llm_dis_map"][i] = None
        state.extras["finish_reason"][i] = "too_long"
        state.extras["prompt_tokens"][i] = 0
        state.extras["gen_tokens"][i] = 0
    for i in np.flatnonzero(skipped.to_numpy()):
        state.generated_texts[i] = SKIPPED_TEXT
        state.extras["llm_answer_raw"][i] = None
        state.extras["llm_dis_map"][i] = NO_MATCH
        state.extras["answer_format_valid"][i] = True
        state.extras["answer_uncertain"][i] = False
        state.extras["answer_ids_in_candidates"][i] = None
        state.extras["finish_reason"][i] = "skipped"
        state.extras["prompt_tokens"][i] = 0
        state.extras["gen_tokens"][i] = 0


def _store_result(state: ShardState, i: int, out: Any, allowed_ids: list[str | None],
                  final_ids: set[str] | None) -> None:
    """Parse one vLLM request output and write its columns into row ``i`` of the state."""
    comp = out.outputs[0]
    text = comp.text
    state.generated_texts[i] = text if text else " "  # non-empty so a resume skips the row
    raw = extract_last_answer(text)
    parsed = parse_answer(raw, allowed_ids)
    state.extras["llm_answer_raw"][i] = raw
    for k, v in parsed.items():
        state.extras[k][i] = v
    state.extras["llm_dis_map_all_panels"][i] = state.extras["llm_dis_map"][i]
    state.extras["llm_dis_map"][i] = restrict_to_panel(state.extras["llm_dis_map"][i], final_ids)
    state.extras["finish_reason"][i] = comp.finish_reason
    state.extras["prompt_tokens"][i] = len(out.prompt_token_ids or [])
    state.extras["gen_tokens"][i] = len(comp.token_ids or [])


def _generate_windows(llm: Any, sampling_params: Any, chat_kwargs: dict[str, Any], df: pd.DataFrame,
                      todo: list[int], window: int, state: ShardState, prep: ShardPrep,
                      res: Resources, out_parquet: str) -> float:
    """Generate the outstanding rows one ``window`` at a time, saving after each window.

    Each window is a single ``llm.chat`` call so the scheduler batches continuously;
    ``save_every`` controls only checkpoint granularity. Returns the generation seconds.
    """
    t_gen = time.time()
    for w_start in range(0, len(todo), window):
        idx = todo[w_start:w_start + window]
        prompts = [df["llm_prompt"].iloc[i] for i in idx]
        logger.info("  generating %d prompt(s) [%d/%d of this shard's outstanding rows]",
                    len(prompts), w_start + len(idx), len(todo))
        conversations = [[{"role": "user", "content": p}] for p in prompts]
        outputs = llm.chat(conversations, sampling_params, use_tqdm=True,
                           chat_template_kwargs=chat_kwargs or None)
        for i, out in zip(idx, outputs):
            _store_result(state, i, out, prep.allowed.iloc[i], res.final_ids)
        _save_progress(df, out_parquet, state)
    return time.time() - t_gen


def _write_run_meta(meta_path: str, settings: dict[str, Any], shard_path: str, out_parquet: str,
                    n_rows: int, todo: list[int], state: ShardState, prep: ShardPrep, res: Resources,
                    too_long_rows: set[int], gen_seconds: float, t_shard: float) -> dict[str, Any]:
    """Write ``run_meta.json`` for one shard: settings plus counts and throughput. Returns it."""
    n_done = len(todo)
    gen_tok = [state.extras["gen_tokens"][i] or 0 for i in todo]
    prm_tok = [state.extras["prompt_tokens"][i] or 0 for i in todo]
    meta = dict(settings)
    meta.update({
        "shard": os.path.basename(shard_path), "out_parquet": out_parquet,
        "rows_total": n_rows, "rows_generated_this_run": n_done,
        "wall_clock_s": round(time.time() - t_shard, 1),
        "generation_s": round(gen_seconds, 1),
        "rows_per_s": round(n_done / gen_seconds, 3) if gen_seconds else None,
        "prompt_tokens_total": int(sum(prm_tok)), "gen_tokens_total": int(sum(gen_tok)),
        "gen_tokens_mean": float(np.mean(gen_tok)) if gen_tok else None,
        "gen_tokens_p95": float(np.percentile(gen_tok, 95)) if gen_tok else None,
        "gen_tokens_per_s": round(sum(gen_tok) / gen_seconds, 1) if gen_seconds else None,
        "truncated_rows": int(sum(1 for i in todo if state.extras["finish_reason"][i] == "length")),
        "no_match_rows": int(sum(1 for i in todo if state.extras["llm_dis_map"][i] == NO_MATCH)),
        "unparsed_rows": int(sum(1 for i in todo if state.extras["llm_dis_map"][i] is None)),
        "hallucinated_rows": int(sum(1 for i in todo
                                     if state.extras["answer_ids_in_candidates"][i] is False)),
        "context_threads_missing": res.context_missing.get("missing", 0),
        "candidates_per_row_mean": float(prep.n_candidates.mean()),
        "candidates_per_row_max": int(prep.n_candidates.max()),
        "rows_skipped_no_candidates": int(prep.skipped.sum()),
        "rows_skipped_too_long": len(too_long_rows),
        # No peak-memory field: vLLM v1 runs the engine in a child process, so the driver's
        # torch.cuda counters read 0; gpu_memory_utilization is the budget.
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------
def run_llm_over_shards(cfg: LlmMapConfig) -> None:
    """Run the adjudication model over every shard parquet in ``cfg.shards_dir``.

    Reads ``*.parquet`` from the shards directory, keeps this worker's row stripe, builds
    the prompts, runs vLLM once per checkpoint window (resumable per shard) and writes the
    output parquet and ``run_meta.json`` described in the module docstring.
    """
    import torch
    from vllm import LLM, SamplingParams

    out_dir = cfg.out_dir or cfg.shards_dir
    os.makedirs(out_dir, exist_ok=True)

    res = _load_resources(cfg)

    sampling_params = SamplingParams(temperature=cfg.temperature, top_p=cfg.top_p,
                                     max_tokens=cfg.max_tokens, seed=cfg.seed)
    # The prompt is a fixed rubric plus a short per-record suffix, so prefix caching
    # avoids re-prefilling the rubric for every record.
    llm_kwargs: dict[str, Any] = {
        "enable_prefix_caching": True,
        "max_num_seqs": cfg.max_num_seqs,
        "max_model_len": cfg.max_model_len,
        "gpu_memory_utilization": cfg.gpu_memory_utilization,
        "seed": cfg.seed,
    }
    if cfg.dtype and cfg.dtype != "auto":
        llm_kwargs["dtype"] = cfg.dtype
    if cfg.tensor_parallel_size is not None:
        llm_kwargs["tensor_parallel_size"] = int(cfg.tensor_parallel_size)
    chat_kwargs: dict[str, Any] = {}
    if cfg.reasoning_effort:
        # GPT-OSS reads this from the chat template ("Reasoning: medium"); models whose
        # template does not use it ignore the variable.
        chat_kwargs["reasoning_effort"] = cfg.reasoning_effort

    t_engine = time.time()
    llm = LLM(model=cfg.llm_model, **llm_kwargs)
    logger.info("Engine up in %.0fs", time.time() - t_engine)

    settings: dict[str, Any] = {
        "stage": "llm_map",
        "model": cfg.llm_model,
        "prompt_file": os.path.abspath(cfg.prompt_file),
        "reasoning_effort": cfg.reasoning_effort,
        "context_json": os.path.abspath(cfg.context_json) if cfg.context_json else None,
        "context_drop_fields": cfg.context_drop_fields,
        "panel_siblings_csv": os.path.abspath(cfg.panel_siblings_csv) if cfg.panel_siblings_csv else None,
        "final_panel_csv": os.path.abspath(cfg.final_panel_csv) if cfg.final_panel_csv else None,
        "temperature": cfg.temperature, "top_p": cfg.top_p, "max_tokens": cfg.max_tokens, "seed": cfg.seed,
        "max_model_len": cfg.max_model_len, "max_num_seqs": cfg.max_num_seqs,
        "gpu_memory_utilization": cfg.gpu_memory_utilization, "dtype": cfg.dtype,
        "tensor_parallel_size": cfg.tensor_parallel_size,
        "shard_index": cfg.shard_index, "num_shards": cfg.num_shards,
        "git_sha": _git_sha(), "image": os.environ.get("LITDD_IMAGE"),
        "versions": _versions(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    shard_paths = sorted(glob.glob(os.path.join(cfg.shards_dir, "*.parquet")))
    logger.info("Found %d parquet shard(s) for this worker.", len(shard_paths))

    for shard_path in shard_paths:
        logger.info("Processing shard: %s", os.path.basename(shard_path))
        t_shard = time.time()
        df = pd.read_parquet(shard_path)
        if cfg.num_shards and cfg.num_shards > 1:
            keep = row_slice_for_worker(len(df), cfg.shard_index, cfg.num_shards)
            df = df.iloc[keep].reset_index(drop=True)
            logger.info("shard %s/%s: rows for this worker: %d", cfg.shard_index, cfg.num_shards, len(df))
        if cfg.limit:
            df = df.iloc[:cfg.limit].reset_index(drop=True)
        if df.empty:
            logger.info("  (no rows for this worker in this file)")
            continue

        prep = _prepare_shard(df, cfg, res)
        too_long_rows = _fit_context_budget(df, cfg, llm.get_tokenizer())

        first_prompt = next((p for p in df["llm_prompt"] if p), "")
        logger.debug("Prompt preview:\n%s", first_prompt[-600:])
        n_rows = len(df)
        logger.info("Total rows in shard: %d", n_rows)

        base = os.path.splitext(os.path.basename(shard_path))[0]
        suffix = "" if cfg.shard_index is None else f"_w{cfg.shard_index}"
        out_parquet = os.path.join(out_dir, f"{base}{suffix}__llm.parquet")
        meta_path = os.path.join(out_dir, f"{base}{suffix}__llm.run_meta.json")

        # Resume from a partially written output when the shard was interrupted.
        state = _load_resume_state(out_parquet, n_rows)
        _mark_skipped_rows(state, prep.skipped, too_long_rows)

        todo = [i for i in range(n_rows) if not state.generated_texts[i]]
        if not todo:
            logger.info("shard already complete")
            continue

        window = cfg.save_every if cfg.save_every and cfg.save_every > 0 else n_rows
        gen_seconds = _generate_windows(llm, sampling_params, chat_kwargs, df, todo, window,
                                        state, prep, res, out_parquet)

        meta = _write_run_meta(meta_path, settings, shard_path, out_parquet, n_rows, todo, state,
                               prep, res, too_long_rows, gen_seconds, t_shard)
        logger.info("Shard completed: %s (%d rows, %s rows/s, %.0f mean gen tokens, %d truncated, %d unparsed)",
                    os.path.basename(shard_path), len(todo), meta["rows_per_s"], meta["gen_tokens_mean"],
                    meta["truncated_rows"], meta["unparsed_rows"])

        torch.cuda.empty_cache()
        gc.collect()

    logger.info("All shards processed.")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> LlmMapConfig:
    """Parse the command line into an ``LlmMapConfig``."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--shards_dir", required=True, type=str,
                   help="Directory of shard parquets from build_llm_shards.py (pmid, tiab, candidates).")
    p.add_argument("--llm_model", required=True, type=str,
                   help="HF id or local path; released run: openai/gpt-oss-20b")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Output directory (default: the shards directory).")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max_tokens", type=int, default=8192,
                   help="Generation budget including the reasoning trace; rows that hit it are "
                        "flagged finish_reason=length and counted in run_meta.json.")
    p.add_argument("--max_model_len", type=int, default=16384)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--dtype", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reasoning_effort", type=str, default="medium",
                   choices=["low", "medium", "high", "none"],
                   help="Passed to the chat template (GPT-OSS). 'none' omits it.")
    p.add_argument("--prompt_file", type=str, default=DEFAULT_PROMPT_FILE,
                   help="Prompt template with {n}, {plural}, {tiab} and {candidate_lines} placeholders.")
    p.add_argument("--context_json", type=str, required=True,
                   help="Contextualised threads from build_context_threads.py, built from the same "
                        "all-panel G2P export as --panel_siblings_csv.")
    p.add_argument("--context_drop_fields", type=str, default=None,
                   help="Comma-separated field labels to remove from contextualised threads, "
                        "e.g. 'Disease Definition,Phenotypes'.")
    p.add_argument("--panel_siblings_csv", type=str, default=None,
                   help="All-panel G2P export: offer every entry of each candidate gene, from "
                        "any panel, so a paper about a non-DD disorder can map there instead of "
                        "to the gene's DD entry.")
    p.add_argument("--final_panel_csv", type=str, default=None,
                   help="Restrict llm_dis_map to this export's ids (the unrestricted answer is "
                        "kept in llm_dis_map_all_panels).")
    p.add_argument("--shard_index", type=int, default=None,
                   help="This worker's index in 0..num_shards-1; rows are striped across workers.")
    p.add_argument("--num_shards", type=int, default=None, help="Number of workers.")
    p.add_argument("--save_every", type=int, default=1000,
                   help="Rows per generation call and checkpoint.")
    p.add_argument("--tensor_parallel_size", type=int, default=None)
    p.add_argument("--max_num_seqs", type=int, default=512, help="vLLM scheduler concurrency.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N rows of each shard (smoke tests).")
    args = p.parse_args(argv)
    fields = {f.name for f in dataclasses.fields(LlmMapConfig)}
    values = {k: v for k, v in vars(args).items() if k in fields}
    values["reasoning_effort"] = None if args.reasoning_effort == "none" else args.reasoning_effort
    return LlmMapConfig(**values)


def main() -> None:
    """Command-line entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    run_llm_over_shards(parse_args())


if __name__ == "__main__":
    main()
