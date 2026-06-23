"""Unit tests for the deterministic logic in `annotate_pubmed/llm_map.py`:
LLM-output parsing, prompt construction, batching and shard selection.

These exercise pure functions only (no vLLM/GPU). Run with
`pytest tests/test_llm_map.py -v`.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "annotate_pubmed"))

import llm_map  # noqa: E402


def test_extract_last_answer_single():
    assert llm_map.extract_last_answer("reasoning...\nANSWER: G2P123") == "G2P123"


def test_extract_last_answer_takes_last_when_multiple():
    txt = "ANSWER: G2P1\n...more reasoning...\nANSWER: G2P9;G2P8"
    assert llm_map.extract_last_answer(txt) == "G2P9;G2P8"


def test_extract_last_answer_no_match_string():
    assert llm_map.extract_last_answer("ANSWER: NO MATCH") == "NO MATCH"


def test_extract_last_answer_none_when_absent_or_empty():
    assert llm_map.extract_last_answer("no answer line here") is None
    assert llm_map.extract_last_answer("") is None
    assert llm_map.extract_last_answer(None) is None


def test_build_llm_prompt_includes_tiab_and_numbered_candidates():
    prompt = llm_map.build_llm_prompt(
        "My TIAB text", ["G2P1 - GENEA - foo", "G2P2 - GENEB - bar"]
    )
    assert "My TIAB text" in prompt
    assert "1) G2P1 - GENEA - foo" in prompt
    assert "2) G2P2 - GENEB - bar" in prompt
    assert "ANSWER:" in prompt  # output schema present


def test_build_llm_prompt_handles_empty_candidates():
    prompt = llm_map.build_llm_prompt("tiab", [])
    assert "tiab" in prompt
    assert "ANSWER:" in prompt


def test_batched_indices():
    assert list(llm_map.batched_indices(0, 5, 2)) == [(0, 2), (2, 4), (4, 5)]
    assert list(llm_map.batched_indices(0, 0, 2)) == []
    assert list(llm_map.batched_indices(3, 7, 10)) == [(3, 7)]


def test_select_shards_for_worker():
    paths = [f"s{i}.parquet" for i in range(6)]
    assert llm_map.select_shards_for_worker(paths, 0, 3) == ["s0.parquet", "s3.parquet"]
    assert llm_map.select_shards_for_worker(paths, 2, 3) == ["s2.parquet", "s5.parquet"]
    # no sharding -> all paths
    assert llm_map.select_shards_for_worker(paths, None, None) == paths
