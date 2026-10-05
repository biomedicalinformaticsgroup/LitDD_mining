"""Tests for the pure-Python helpers in ``litdd.training.screen_common``."""
from __future__ import annotations

import csv
import hashlib

import numpy as np
import pandas as pd
import pytest

from litdd.training import screen_common as sc


# ----------------------------- gene_fold ------------------------------------ #
def test_gene_fold_is_deterministic_and_md5_based():
    expected = int(hashlib.md5(b"ATAD3A").hexdigest(), 16) % 10
    assert sc.gene_fold("ATAD3A", 10) == expected
    assert sc.gene_fold("ATAD3A", 10) == sc.gene_fold("ATAD3A", 10)


@pytest.mark.parametrize("modulus", [5, 10])
def test_gene_fold_respects_modulus(modulus):
    genes = [f"GENE{i}" for i in range(200)]
    folds = {sc.gene_fold(g, modulus) for g in genes}
    assert folds <= set(range(modulus))
    assert len(folds) == modulus  # 200 genes cover every bucket


def test_gene_fold_of_five_and_ten_agree_on_bucket_zero_subset():
    # Every gene in bucket 0 of 10 is also in bucket 0 of 5 (10 is a multiple of 5).
    genes = [f"G{i}" for i in range(500)]
    for g in genes:
        if sc.gene_fold(g, 10) == 0:
            assert sc.gene_fold(g, 5) == 0


def test_fold_name():
    genes = [f"G{i}" for i in range(100)]
    for g in genes:
        expected = "heldout" if sc.gene_fold(g, 5) == 0 else "train"
        assert sc.fold_name(g, 5) == expected


# ------------------------- confirm_positive parsing ------------------------- #
@pytest.mark.parametrize("token,expected", [
    ("1", 1), ("yes", 1), ("true", 1), ("y", 1), (" YES ", 1), ("True", 1),
    ("0", 0), ("no", 0), ("false", 0), ("n", 0), (" N ", 0),
    ("", None), ("maybe", None), ("2", None), (None, None), (float("nan"), None),
])
def test_parse_confirm_positive(token, expected):
    assert sc.parse_confirm_positive(token) == expected


def test_confirmed_worksheet_filters_labels_and_joins_tiab():
    ws = pd.DataFrame({
        "pmid": ["1", "2", "3", "4"],
        "title": ["T1", "T2 ", "T3", "T4"],
        "abstract": ["A1", " A2", "", "A4"],
        "confirm_positive": ["yes", "0", "", "N"],
        "g2p_id": ["G2P1", "G2P2", "G2P3", "G2P4"],
    })
    out = sc.confirmed_worksheet(ws)
    assert list(out["pmid"]) == ["1", "2", "4"]
    assert list(out["label"]) == [1, 0, 0]
    assert out["label"].dtype.kind == "i"
    assert list(out["tiab"]) == ["T1 A1", "T2   A2", "T4 A4"]
    # The input frame is not modified.
    assert "label" not in ws.columns


# -------------------------- grid parsing ------------------------------------ #
def test_parse_floats_and_ints():
    assert sc.parse_floats(["1e-5", "3e-5", "0.3"]) == [1e-5, 3e-5, 0.3]
    assert sc.parse_ints(["3", "5"]) == [3, 5]
    assert sc.parse_floats([]) == []
    with pytest.raises(ValueError):
        sc.parse_ints(["3.5"])


# -------------------------- confusion counts -------------------------------- #
def test_cm_counts_with_numpy_inputs():
    preds = np.array([1, 1, 0, 0, 1, 0])
    labels = np.array([1, 0, 1, 0, 1, 0])
    assert sc.cm(preds, labels) == {"tp": 2, "fp": 1, "fn": 1, "tn": 2}


def test_cm_accepts_lists_and_returns_python_ints():
    out = sc.cm([1, 0], [1, 1])
    assert out == {"tp": 1, "fp": 0, "fn": 1, "tn": 0}
    assert all(type(v) is int for v in out.values())


# ------------------------------ CSV helpers --------------------------------- #
def test_append_csv_row_and_load_existing(tmp_path):
    path = tmp_path / "results.csv"
    assert sc.load_existing(str(path)) == set()
    sc.append_csv_row(str(path), {"model": "a", "seed": 42, "f1": 0.5})
    sc.append_csv_row(str(path), {"model": "b", "seed": 43, "f1": 0.6})
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["model"] for r in rows] == ["a", "b"]
    assert sc.load_existing(str(path)) == {("a", "42"), ("b", "43")}


def test_maybe_load_hp_json(tmp_path):
    assert sc.maybe_load_hp_json(None) == {}
    p = tmp_path / "hp.json"
    p.write_text('{"best": {"learning_rate": 1e-5}, "results": []}')
    assert sc.maybe_load_hp_json(str(p)) == {"learning_rate": 1e-5}
    p.write_text('{"learning_rate": 3e-5}')
    assert sc.maybe_load_hp_json(str(p)) == {"learning_rate": 3e-5}
