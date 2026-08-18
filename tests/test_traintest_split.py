"""Unit test for the gene/disease-held-out grouping helper used by the stricter
train/test splits (Reviewer 2 E1/E2). Pure pandas; run with
`pytest tests/test_traintest_split.py -v`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "litdd" / "training"))

import final_traintest_dataset as ftd  # noqa: E402


def test_derive_group_columns_parses_gene_and_disease():
    df = pd.DataFrame({
        "g2p_lgmde": [
            "G2P01760 - ATAD3A - 612316 - HGNC:25567 - FLJ10709 - ATAD3A-related disorder - x",
            "G2P00700 - ATP7A - 300011 - HGNC:869 - MNK - ATP7A-related Menkes disease - y",
        ],
        "label": [0, 1],
    })
    out = ftd.derive_group_columns(df)
    assert list(out["g2p_id"]) == ["G2P01760", "G2P00700"]
    assert list(out["gene"]) == ["ATAD3A", "ATP7A"]
    # original columns preserved
    assert "label" in out.columns and "g2p_lgmde" in out.columns


def test_derive_group_columns_handles_malformed():
    df = pd.DataFrame({"g2p_lgmde": ["G2P99999", ""], "label": [0, 0]})
    out = ftd.derive_group_columns(df)
    assert out["g2p_id"].iloc[0] == "G2P99999"
    assert out["gene"].iloc[0] == ""   # no gene field present
    assert out["gene"].iloc[1] == ""
