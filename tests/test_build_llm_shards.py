"""Unit tests for ``litdd.pipeline.build_llm_shards`` on an inline polars frame."""
from __future__ import annotations

import os

import polars as pl
import pytest

from litdd.pipeline import build_llm_shards


def _candidates_frame() -> pl.DataFrame:
    return pl.DataFrame({
        "pmid": ["p1", "p2", "p3", "p4", "p5"],
        "tiab": ["t1", "t2", "t3", "t4", "t5"],
        "candidate_g2p_ids": [["G2P1", "G2P2"], [], ["G2P3"], ["G2P4", "G2P1", "G2P5"], []],
        "candidate_sources": [["sym", "sym"], [], ["hgnc"], ["sym", "hgnc", "sym"], []],
        "extra": [1, 2, 3, 4, 5],
    })


def test_shard_frame_schema_and_dropped_rows():
    out = build_llm_shards.shard_frame(_candidates_frame())
    assert out.columns == ["pmid", "tiab", "candidates"]
    assert out.schema["candidates"] == pl.List(pl.Utf8)
    assert out.height == 3
    assert out["pmid"].to_list() == ["p1", "p3", "p4"]


def test_write_shards_single_file_preserves_every_id(tmp_path):
    paths = build_llm_shards.write_shards(_candidates_frame(), str(tmp_path), num_shards=1)
    assert paths == [os.path.join(str(tmp_path), "corpus_shard0-of-1.parquet")]
    shard = pl.read_parquet(paths[0])
    assert shard.columns == ["pmid", "tiab", "candidates"]
    assert shard.schema["candidates"] == pl.List(pl.Utf8)
    assert shard.height == 3
    assert shard["candidates"].to_list() == [["G2P1", "G2P2"], ["G2P3"], ["G2P4", "G2P1", "G2P5"]]
    assert shard["tiab"].to_list() == ["t1", "t3", "t4"]


def test_write_shards_by_rows_per_shard(tmp_path):
    paths = build_llm_shards.write_shards(_candidates_frame(), str(tmp_path), rows_per_shard=2, stem="x")
    assert [os.path.basename(p) for p in paths] == ["x_shard0-of-2.parquet", "x_shard1-of-2.parquet"]
    parts = [pl.read_parquet(p) for p in paths]
    assert [p.height for p in parts] == [2, 1]
    assert pl.concat(parts)["pmid"].to_list() == ["p1", "p3", "p4"]


def test_write_shards_by_num_shards(tmp_path):
    paths = build_llm_shards.write_shards(_candidates_frame(), str(tmp_path), num_shards=3)
    assert len(paths) == 3
    parts = [pl.read_parquet(p) for p in paths]
    assert [p.height for p in parts] == [1, 1, 1]
    ids = [c for p in parts for row in p["candidates"].to_list() for c in row]
    assert ids == ["G2P1", "G2P2", "G2P3", "G2P4", "G2P1", "G2P5"]
    # more shards requested than rows: the count is capped at the row count
    paths = build_llm_shards.write_shards(_candidates_frame(), str(tmp_path / "b"), num_shards=10)
    assert len(paths) == 3 and paths[0].endswith("corpus_shard0-of-3.parquet")


def test_write_shards_requires_exactly_one_split_flag(tmp_path):
    with pytest.raises(ValueError):
        build_llm_shards.write_shards(_candidates_frame(), str(tmp_path))
    with pytest.raises(ValueError):
        build_llm_shards.write_shards(_candidates_frame(), str(tmp_path), num_shards=1, rows_per_shard=1)


def test_write_shards_with_no_candidate_rows_writes_nothing(tmp_path):
    df = _candidates_frame().filter(pl.col("candidate_g2p_ids").list.len() == 0)
    assert build_llm_shards.write_shards(df, str(tmp_path / "empty"), num_shards=2) == []


def test_main_reads_parquet_and_defaults_to_one_shard(tmp_path):
    src = tmp_path / "candidates.parquet"
    _candidates_frame().write_parquet(src)
    out = tmp_path / "shards"
    build_llm_shards.main(["--candidates_parquet", str(src), "--out_dir", str(out)])
    files = sorted(os.listdir(out))
    assert files == ["corpus_shard0-of-1.parquet"]
    assert pl.read_parquet(out / files[0]).height == 3
