"""Tests for the shared G2P LGMDE thread builder (`litdd/threads.py`).

The thread is the exact string the cross-encoder is fine-tuned on, so these guard the
properties that a train/inference divergence would break.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from litdd.threads import LGMDE_FIELDS, build_lgmde_list, build_lgmde_map  # noqa: E402

CSV = textwrap.dedent("""\
    g2p id,gene symbol,gene mim,hgnc id,previous gene symbols,disease name,disease mim,disease MONDO,allelic requirement,cross cutting modifier,confidence,variant consequence,variant types,molecular mechanism,molecular mechanism support,panel
    G2P00410,NF1,613113,7765,,NF1-related neurofibromatosis,162200,,monoallelic_autosomal,,definitive,absent gene product,,loss of function,inferred,DD
    G2P00001,HMX1,142992,5017,H6,HMX1-related oculoauricular syndrome,612109,MONDO:0012802,biallelic_autosomal,,limited,altered gene product structure,missense_variant,loss of function,evidence,DD
    G2P09999,TEST1,,9999,,TEST1-related disorder,,,,,,,,,,DD
    """)


@pytest.fixture()
def g2p_csv(tmp_path):
    p = tmp_path / "g2p.csv"
    p.write_text(CSV)
    return str(p)


def test_thread_has_all_fields_and_is_stable(g2p_csv):
    m = build_lgmde_map(g2p_csv)
    assert set(m) == {"G2P00410", "G2P00001", "G2P09999"}
    for thread in m.values():
        assert len(thread.split(" - ")) == len(LGMDE_FIELDS)


def test_variant_consequence_is_populated(g2p_csv):
    """The regression this module exists for.

    Inference used to look for a column named "inferred variant consequence", which no G2P
    export has, so this field was blank in every candidate while training had it filled.
    """
    thread = build_lgmde_map(g2p_csv)["G2P00410"]
    assert "absent gene product" in thread
    fields = thread.split(" - ")
    idx = [f for f, _ in LGMDE_FIELDS].index("variant_consequence")
    assert fields[idx] == "absent gene product"


def test_reproduces_training_rendering(g2p_csv):
    """Missing -> 'nan', numeric MIMs keep the float form, HGNC keeps its prefix."""
    thread = build_lgmde_map(g2p_csv)["G2P00410"]
    assert thread == (
        "G2P00410 - NF1 - 613113.0 - HGNC:7765 - nan - NF1-related neurofibromatosis"
        " - 162200.0 - nan - monoallelic_autosomal - nan - definitive"
        " - absent gene product - nan - loss of function - inferred"
    )


def test_2025_export_layout_renders_the_annotated_set_threads(tmp_path):
    """The 2025-02-15 export (the panel the annotated set was labelled against) stores the
    HGNC id as a prefixed string and calls the last field 'molecular mechanism
    categorisation'. It must render to the same 15-field string as the annotated set --
    no 'HGNC:HGNC:' and no missing-column error -- so the LLM evaluation can re-render the
    same panel the labels refer to."""
    p = tmp_path / "g2p_2025.csv"
    p.write_text(textwrap.dedent("""\
        g2p id,gene symbol,gene mim,hgnc id,previous gene symbols,disease name,disease mim,disease MONDO,allelic requirement,cross cutting modifier,confidence,inferred variant consequence,variant types,molecular mechanism,molecular mechanism categorisation,molecular mechanism evidence,panel
        G2P00117,COL11A2,120290,HGNC:2187,DFNA13; DFNB53; HKE5,COL11A2-related otospondylomegaepiphyseal dysplasia,215150,,biallelic_autosomal,restricted mutation set,definitive,altered gene product structure,,dominant negative,inferred,,DD
        G2P09999,TEST1,,HGNC:9999,,TEST1-related disorder,,,,,,,,,,,DD
        """))
    # (the blank MIMs on the second row give the columns the float dtype the real export
    # has, which is what produces the "120290.0" rendering the annotated set carries)
    thread = build_lgmde_map(str(p))["G2P00117"]
    assert thread == (
        "G2P00117 - COL11A2 - 120290.0 - HGNC:2187 - DFNA13; DFNB53; HKE5 - "
        "COL11A2-related otospondylomegaepiphyseal dysplasia - 215150.0 - nan - "
        "biallelic_autosomal - restricted mutation set - definitive - "
        "altered gene product structure - nan - dominant negative - inferred"
    )


def test_missing_column_raises_rather_than_blanking(tmp_path):
    """A renamed/absent column must fail loudly, not silently emit an empty field."""
    p = tmp_path / "bad.csv"
    p.write_text("g2p id,gene symbol\nG2P00410,NF1\n")
    with pytest.raises(KeyError, match="missing column"):
        build_lgmde_map(str(p))


def test_rendering_is_dtype_dependent(g2p_csv):
    """Documents a real fragility rather than hiding it.

    A MIM column containing any blank is read as float64, so 613113 renders as "613113.0";
    an all-integer column renders as "613113". The thread string therefore depends on other
    rows' missingness. Real G2P exports always contain blanks, which is what the released
    cross-encoder was fine-tuned against -- so this must not be "tidied" without re-finetuning.
    """
    assert "613113.0" in build_lgmde_map(g2p_csv)["G2P00410"]


def test_list_is_unique_and_sorted(g2p_csv):
    threads = build_lgmde_list(g2p_csv)
    assert threads == sorted(set(threads))
