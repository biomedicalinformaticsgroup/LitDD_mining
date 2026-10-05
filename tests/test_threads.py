"""Tests for the G2P entry label renderer in litdd.threads."""
from __future__ import annotations

import textwrap

import pytest

from litdd.threads import LGMDE_FIELDS, build_lgmde_list, build_lgmde_map

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


def test_label_has_every_field(g2p_csv):
    m = build_lgmde_map(g2p_csv)
    assert set(m) == {"G2P00410", "G2P00001", "G2P09999"}
    for label in m.values():
        assert len(label.split(" - ")) == len(LGMDE_FIELDS)


def test_variant_consequence_is_populated(g2p_csv):
    label = build_lgmde_map(g2p_csv)["G2P00410"]
    fields = label.split(" - ")
    idx = [f for f, _ in LGMDE_FIELDS].index("variant_consequence")
    assert fields[idx] == "absent gene product"


def test_rendering_of_missing_values_and_hgnc_prefix(g2p_csv):
    """Blanks render as 'nan', numeric columns with blanks keep the float form, the HGNC id
    carries its prefix."""
    label = build_lgmde_map(g2p_csv)["G2P00410"]
    assert label == (
        "G2P00410 - NF1 - 613113.0 - HGNC:7765 - nan - NF1-related neurofibromatosis"
        " - 162200.0 - nan - monoallelic_autosomal - nan - definitive"
        " - absent gene product - nan - loss of function - inferred"
    )


def test_earlier_export_layout_renders_the_same_fields(tmp_path):
    """An export that stores the HGNC id with its prefix and names the last field
    'molecular mechanism categorisation' renders to the same fifteen fields."""
    p = tmp_path / "g2p_2025.csv"
    p.write_text(textwrap.dedent("""\
        g2p id,gene symbol,gene mim,hgnc id,previous gene symbols,disease name,disease mim,disease MONDO,allelic requirement,cross cutting modifier,confidence,inferred variant consequence,variant types,molecular mechanism,molecular mechanism categorisation,molecular mechanism evidence,panel
        G2P00117,COL11A2,120290,HGNC:2187,DFNA13; DFNB53; HKE5,COL11A2-related otospondylomegaepiphyseal dysplasia,215150,,biallelic_autosomal,restricted mutation set,definitive,altered gene product structure,,dominant negative,inferred,,DD
        G2P09999,TEST1,,HGNC:9999,,TEST1-related disorder,,,,,,,,,,,DD
        """))
    label = build_lgmde_map(str(p))["G2P00117"]
    assert label == (
        "G2P00117 - COL11A2 - 120290.0 - HGNC:2187 - DFNA13; DFNB53; HKE5 - "
        "COL11A2-related otospondylomegaepiphyseal dysplasia - 215150.0 - nan - "
        "biallelic_autosomal - restricted mutation set - definitive - "
        "altered gene product structure - nan - dominant negative - inferred"
    )


def test_missing_column_raises(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("g2p id,gene symbol\nG2P00410,NF1\n")
    with pytest.raises(KeyError, match="missing column"):
        build_lgmde_map(str(p))


def test_rendering_depends_on_column_dtype(g2p_csv):
    """A numeric column containing a blank is read as float, so 613113 renders as 613113.0."""
    assert "613113.0" in build_lgmde_map(g2p_csv)["G2P00410"]


def test_list_is_unique_and_sorted(g2p_csv):
    labels = build_lgmde_list(g2p_csv)
    assert labels == sorted(set(labels))
