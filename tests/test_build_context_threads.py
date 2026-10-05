"""Tests for the contextualised-thread builder (`litdd/pipeline/build_context_threads.py`).

A three-row G2P export, two minimal OBO files and a two-line alias table exercise the block
layout, the MONDO and HPO enrichment, the previous-symbol relabelling and the provenance entry.
"""
from __future__ import annotations

import json
import textwrap

import pandas as pd
import pytest

from litdd.pipeline.build_context_threads import (
    PREVIOUS_LABEL_IN,
    PREVIOUS_LABEL_OUT,
    build_context_threads,
    main,
    relabel_previous_symbols,
)

G2P_CSV = textwrap.dedent("""\
    g2p id,gene symbol,gene mim,hgnc id,previous gene symbols,disease name,disease mim,disease MONDO,allelic requirement,cross cutting modifier,confidence,variant consequence,variant types,molecular mechanism,molecular mechanism support,molecular mechanism categorisation,molecular mechanism evidence,phenotypes,publications,additional mined publications,panel,comments,date of last review,review
    G2P00001,HMX1,142992,5017,H6; NKX5-3,HMX1-related oculoauricular syndrome,612109,MONDO:0000002,biallelic_autosomal,,definitive,absent gene product,,loss of function,inferred,,,HP:0000002; HP:0000003,18423520,21417677,DD; Eye,,2019-09-26 16:23:46+00:00,
    G2P00002,NOTCH3,600276,7883,CADASIL; IMF2,NOTCH3-related CADASIL,125310,,monoallelic_autosomal,,definitive,altered gene product structure,,dominant negative,evidence,,,HP:0000003,,,DD,,2020-01-01 00:00:00+00:00,
    G2P00003,TEST1,,9999,,TEST1-related disorder,,,,,limited,,,,,,,,,,DD,,,
    """)

MONDO_OBO = textwrap.dedent("""\
    format-version: 1.2
    data-version: releases/2030-01-01
    synonymtypedef: ABBREVIATION "abbreviation"
    ontology: mondo

    [Term]
    id: MONDO:0000001
    name: disease

    [Term]
    id: MONDO:0000002
    name: oculoauricular syndrome
    def: "A rare developmental defect of the eye and outer ear." [Orphanet:1234]
    synonym: "OCACS" EXACT ABBREVIATION []
    synonym: "oculo-auricular syndrome" RELATED []
    synonym: "HMX1-related oculoauricular syndrome" RELATED []
    is_a: MONDO:0000001 ! disease
    """)

HP_OBO = textwrap.dedent("""\
    format-version: 1.2
    data-version: hp/releases/2030-02-02
    ontology: hp

    [Term]
    id: HP:0000001
    name: All

    [Term]
    id: HP:0000002
    name: Microcornea
    def: "A cornea that is smaller than usual." [HPO:probinson]
    is_a: HP:0000001 ! All

    [Term]
    id: HP:0000003
    name: Cataract
    synonym: "Lens opacity" EXACT []
    is_a: HP:0000001 ! All
    """)

ALIASES_TSV = "kind\tgene\talias\tsource\tmondo_example\nsymbol\tNOTCH3\tCADASIL\thgnc_alias_symbol\tCADASIL\n"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("ctx")
    (d / "G2P_test.csv").write_text(G2P_CSV)
    (d / "mondo.obo").write_text(MONDO_OBO)
    (d / "hp.obo").write_text(HP_OBO)
    (d / "aliases.tsv").write_text(ALIASES_TSV)
    threads, provenance, enriched = build_context_threads(
        d / "G2P_test.csv", d / "mondo.obo", d / "hp.obo", d / "aliases.tsv"
    )
    return d, threads, provenance, enriched


def test_block_layout_with_mondo_and_hpo(built):
    _, threads, _, _ = built
    assert threads["G2P00001"].splitlines() == [
        "G2P ID: G2P00001",
        "Gene Symbol: HMX1",
        "Disease Name: HMX1-related oculoauricular syndrome",
        f"{PREVIOUS_LABEL_OUT} H6; NKX5-3",
        "Disease Synonyms: OCACS; oculo-auricular syndrome; oculoauricular syndrome",
        "Disease Definition: A rare developmental defect of the eye and outer ear.",
        "Phenotypes: Microcornea; Cataract",
        "Allelic Requirement: biallelic_autosomal",
        "Molecular Mechanism: loss of function",
        "Variant Consequence: absent gene product",
    ]


def test_disease_alias_removed_and_line_relabelled(built):
    _, threads, _, _ = built
    lines = threads["G2P00002"].splitlines()
    assert f"{PREVIOUS_LABEL_OUT} IMF2" in lines
    assert not any(line.startswith(PREVIOUS_LABEL_IN) for line in lines)
    assert "CADASIL" not in threads["G2P00002"].split("Disease Name:")[1].split("\n")[1]
    assert "Disease Synonyms: None" in lines
    assert "Disease Definition: None" in lines
    assert "Phenotypes: Cataract" in lines


def test_blank_fields_render_as_none_and_drop_previous_line(built):
    _, threads, _, _ = built
    lines = threads["G2P00003"].splitlines()
    assert len(lines) == 9
    assert not any(line.startswith("Previous Gene Symbols") for line in lines)
    for label in ("Disease Synonyms", "Disease Definition", "Phenotypes", "Allelic Requirement",
                  "Molecular Mechanism", "Variant Consequence"):
        assert f"{label}: None" in lines


def test_relabel_drops_line_when_every_symbol_is_an_alias():
    block = "G2P ID: X\nGene Symbol: MECP2\nPrevious Gene Symbols: RTT; rtt; None\nPhenotypes: None\n"
    out = relabel_previous_symbols(block, "MECP2", {"MECP2": {"RTT"}})
    assert out == "G2P ID: X\nGene Symbol: MECP2\nPhenotypes: None"
    kept = relabel_previous_symbols(block, "MECP2", {})
    assert kept.splitlines()[2] == f"{PREVIOUS_LABEL_OUT} RTT; rtt"


def test_provenance_and_enrichment(built):
    d, threads, provenance, enriched = built
    assert list(threads) == ["G2P00001", "G2P00002", "G2P00003"]
    assert provenance["panel"] == "G2P_test.csv"
    assert provenance["mondo"] == "releases/2030-01-01"
    assert provenance["hpo"] == "hp/releases/2030-02-02"
    assert provenance["format"] == "v1"
    assert provenance["aliases"] == "aliases.tsv"
    assert provenance["unresolved_mondo_ids"] == [] and provenance["unresolved_hpo_ids"] == []
    assert str(d) not in json.dumps(provenance)
    assert enriched.loc[0, "hpo_term_names"] == ["Microcornea", "Cataract"]
    assert pd.isna(enriched.loc[2, "mondo_name"])


def test_cli_writes_json_and_parquet(built):
    d, _, _, _ = built
    out_json = d / "out" / "threads.json"
    out_parquet = d / "out" / "threads.parquet"
    rc = main([
        "--g2p_csv", str(d / "G2P_test.csv"), "--mondo_obo", str(d / "mondo.obo"), "--hp_obo", str(d / "hp.obo"),
        "--aliases_tsv", str(d / "aliases.tsv"), "--out_json", str(out_json), "--keep_parquet", str(out_parquet),
    ])
    assert rc == 0
    data = json.loads(out_json.read_text())
    assert list(data) == ["__provenance__", "G2P00001", "G2P00002", "G2P00003"]
    assert data["G2P00001"].startswith("G2P ID: G2P00001\nGene Symbol: HMX1\n")
    assert out_parquet.exists()
