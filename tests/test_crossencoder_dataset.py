"""Tests for the cross-encoder dataset rebuild (`litdd/training/build_crossencoder_dataset.py`)
and the thread-variant rendering it depends on (`litdd/threads.py` gene-name support).

These guard the leak-control and provenance properties of the re-finetune:
  * re-rendering swaps stale thread text for the current export's rendering, by g2p_id;
  * rows whose G2P entry was retired are dropped, not silently kept with stale text;
  * gene-level augmentation positives survive only for single-entry genes;
  * the flat rendering stays byte-identical when no gene_names dict is passed (the
    released cross-encoder was fine-tuned on it).
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from litdd.threads import build_lgmde_map, load_gene_names  # noqa: E402
from litdd.training.build_crossencoder_dataset import (  # noqa: E402
    filter_aug_rows,
    panel_gene_info,
    parse_g2p_id,
    pmid_set,
    rerender,
    tiab_to_pmids,
)

CSV = textwrap.dedent("""\
    g2p id,gene symbol,gene mim,hgnc id,previous gene symbols,disease name,disease mim,disease MONDO,allelic requirement,cross cutting modifier,confidence,variant consequence,variant types,molecular mechanism,molecular mechanism support,panel
    G2P00410,NF1,613113,7765,,NF1-related neurofibromatosis,162200,,monoallelic_autosomal,,definitive,absent gene product,,loss of function,inferred,DD
    G2P00001,AFP,142992,317,H6,AFP-related oculoauricular syndrome,612109,MONDO:0012802,biallelic_autosomal,,limited,altered gene product structure,missense_variant,loss of function,evidence,DD
    G2P00002,AFP,142992,317,,AFP-related second disorder,,,monoallelic_autosomal,,limited,,,,,DD
    G2P09999,GALNS,,4122,,GALNS-related disorder,,,,,,,,,,DD
    G2P08888,NOTAREALGENE,,,,NOTAREALGENE-related disorder,,,,,,,,,,DD
    """)


@pytest.fixture()
def g2p_csv(tmp_path):
    p = tmp_path / "g2p.csv"
    p.write_text(CSV)
    return str(p)


GENE_INFO = str(ROOT / "tests" / "fixtures" / "gene_info_sample.gz")


def test_gene_name_variant_inserts_after_previous_symbols(g2p_csv):
    flat = build_lgmde_map(g2p_csv)
    named = build_lgmde_map(g2p_csv, gene_names=load_gene_names(GENE_INFO))
    # flat rendering untouched: 15 fields; gene-name variant adds exactly one
    assert all(len(t.split(" - ")) == 15 for t in flat.values())
    assert all(len(t.split(" - ")) == 16 for t in named.values())
    # inserted directly after previous gene symbols (field index 4 -> name at 5)
    fields = named["G2P00001"].split(" - ")
    assert fields[4] == "H6"
    assert fields[5] == "alpha fetoprotein"       # resolved via HGNC id 317
    assert named["G2P09999"].split(" - ")[5] == "galactosamine (N-acetyl)-6-sulfatase"
    assert named["G2P00410"].split(" - ")[5] == "neurofibromin 1"   # via symbol NF1
    # a gene absent from gene_info renders an empty field, never a crash
    assert named["G2P08888"].split(" - ")[5] == ""


def test_rerender_swaps_thread_text_and_drops_retired_ids(g2p_csv):
    thread_map = build_lgmde_map(g2p_csv)
    df = pd.DataFrame({
        "tiab": ["paper A", "paper B", "paper C"],
        "g2p_lgmde": [
            "G2P00410 - NF1 - stale - old rendering",   # entry still in panel
            "G2P00001 - AFP - stale - old rendering",
            "G2P07777 - GONE - retired entry",          # not in panel any more
        ],
        "label": [1, 0, 1],
    })
    out, dropped = rerender(df, thread_map)
    assert len(out) == 2 and len(dropped) == 1
    assert dropped.iloc[0]["g2p_id"] == "G2P07777"
    assert out.iloc[0]["g2p_lgmde"] == thread_map["G2P00410"]   # current text, not stale
    assert list(out["label"]) == [1, 0]                          # labels untouched


def test_parse_g2p_id():
    assert parse_g2p_id("G2P00410 - NF1 - x") == "G2P00410"
    assert parse_g2p_id("G2P00410") == "G2P00410"


def test_aug_filter_keeps_single_entry_genes_only(g2p_csv):
    id2gene, counts = panel_gene_info(g2p_csv)
    assert id2gene["G2P00001"] == "AFP"
    assert counts["G2P00001"] == 2 and counts["G2P00410"] == 1
    thread_map = build_lgmde_map(g2p_csv)
    df = pd.DataFrame({
        "tiab": ["aug single", "aug multi", "aug neg", "normal"],
        "g2p_lgmde": [thread_map["G2P00410"], thread_map["G2P00001"],
                      thread_map["G2P00002"], thread_map["G2P00001"]],
        "label": [1, 1, 0, 1],
    })
    aug_keys = {("aug single", "G2P00410"), ("aug multi", "G2P00001"),
                ("aug neg", "G2P00002"), ("shared paper", "G2P00001")}
    df.loc[len(df)] = ["shared paper", thread_map["G2P00002"], 1]  # external row, same
    # paper as an aug key but a DIFFERENT g2p_id -> must not be dropped
    out, n_aug_pos, n_dropped = filter_aug_rows(df, aug_keys, counts)
    assert n_aug_pos == 2 and n_dropped == 1
    kept = set(out["tiab"])
    assert "aug single" in kept          # single-entry gene positive survives
    assert "aug multi" not in kept       # multi-entry gene positive dropped
    assert "aug neg" in kept             # aug negatives are never dropped
    assert "normal" in kept              # non-aug rows untouched
    assert "shared paper" in kept        # external pair for an aug paper untouched


def test_pmid_disjointness_helpers(tmp_path):
    src = tmp_path / "src.csv"
    pd.DataFrame({"pmid": ["1", "2"], "tiab": ["text one", "text two"]}).to_csv(
        src, index=False)
    t2p = tiab_to_pmids([str(src)])
    pmids, unmapped = pmid_set({"text one", "unknown text"}, t2p)
    assert pmids == {"1"} and unmapped == 1
