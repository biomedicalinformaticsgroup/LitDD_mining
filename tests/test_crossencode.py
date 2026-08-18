"""Unit tests for the deterministic ranking logic in `litdd/pipeline/crossencode.py`:
top-5 candidate selection (min-heap update), pair construction, and the G2P LGMDE
string builder.

These exercise pure functions only (no torch/sentence-transformers/GPU). Run with
`pytest tests/test_crossencode.py -v`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "litdd" / "pipeline"))

import crossencode  # noqa: E402


def test_make_pairs_for_block_order():
    pairs = crossencode.make_pairs_for_block(["t0", "t1"], ["g0", "g1", "g2"])
    assert pairs == [
        ("t0", "g0"), ("t0", "g1"), ("t0", "g2"),
        ("t1", "g0"), ("t1", "g1"), ("t1", "g2"),
    ]


def test_update_topk_heaps_selects_top_k():
    heaps = [[]]
    scores = np.array([[0.1, 0.9, 0.5, 0.7, 0.2]])
    g_block = ["a", "b", "c", "d", "e"]
    crossencode.update_topk_heaps_from_block(heaps, scores, g_block, k=3)
    top = sorted(heaps[0], key=lambda x: x[0], reverse=True)
    assert [lbl for _, lbl in top] == ["b", "d", "c"]  # 0.9, 0.7, 0.5
    assert len(heaps[0]) == 3  # heap holds exactly k


def test_update_topk_heaps_merges_across_blocks():
    heaps = [[]]
    crossencode.update_topk_heaps_from_block(heaps, np.array([[0.4, 0.3]]), ["a", "b"], k=2)
    crossencode.update_topk_heaps_from_block(heaps, np.array([[0.9, 0.1]]), ["c", "d"], k=2)
    top = sorted(heaps[0], key=lambda x: x[0], reverse=True)
    assert [lbl for _, lbl in top] == ["c", "a"]  # 0.9 (block2), 0.4 (block1)


def test_update_topk_heaps_k_larger_than_candidates():
    heaps = [[]]
    crossencode.update_topk_heaps_from_block(heaps, np.array([[0.5, 0.6]]), ["a", "b"], k=5)
    assert len(heaps[0]) == 2  # cannot keep more than available


def test_update_topk_heaps_independent_rows():
    heaps = [[], []]
    scores = np.array([[0.9, 0.1], [0.2, 0.8]])
    crossencode.update_topk_heaps_from_block(heaps, scores, ["a", "b"], k=1)
    assert heaps[0][0][1] == "a"  # best for row 0
    assert heaps[1][0][1] == "b"  # best for row 1


def test_build_g2p_lgmde_list(tmp_path):
    csv_path = tmp_path / "g2p.csv"
    csv_path.write_text(
        "g2p id,gene symbol,disease name,allelic requirement,confidence\n"
        "G2P1,GENEA,Disease A,monoallelic,definitive\n"
        "G2P2,GENEB,Disease B,biallelic,limited\n"
    )
    out = crossencode.build_g2p_lgmde_list(str(csv_path))
    assert len(out) == 2
    joined = "\n".join(out)
    assert "G2P1 - GENEA" in joined
    assert "Disease A" in joined and "monoallelic" in joined
    assert "G2P2 - GENEB" in joined


def test_build_g2p_lgmde_list_dedups_identical_rows(tmp_path):
    csv_path = tmp_path / "g2p_dup.csv"
    csv_path.write_text(
        "g2p id,gene symbol,disease name\n"
        "G2P1,GENEA,Disease A\n"
        "G2P1,GENEA,Disease A\n"
    )
    out = crossencode.build_g2p_lgmde_list(str(csv_path))
    assert len(out) == 1  # identical concatenated rows collapse to one unique entry
