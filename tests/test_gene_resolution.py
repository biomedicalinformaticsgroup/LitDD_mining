"""Unit tests for the MONDO disease lexicon and the HGNC panel (``litdd/gene_resolution.py``)."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from litdd.gene_resolution import DiseaseLexicon, HgncPanel, resolve_candidates
from litdd.genes import normalise_phrase

MONDO_OBO = textwrap.dedent("""\
    format-version: 1.2

    [Term]
    id: MONDO:0000001
    name: neurodevelopmental disorder

    [Term]
    id: MONDO:0008434
    name: Smith-Magenis syndrome
    synonym: "SMS" EXACT []
    synonym: "Smith Magenis syndrome" EXACT []
    synonym: "SMS syndrome" RELATED []
    is_a: MONDO:0000001
    relationship: has_material_basis_in_germline_mutation_in http://identifiers.org/hgnc/9791 ! RAI1

    [Term]
    id: MONDO:0010679
    name: Duchenne muscular dystrophy
    synonym: "DMD" EXACT []
    intersection_of: has_material_basis_in_germline_mutation_in http://identifiers.org/hgnc/2928 ! DMD

    [Term]
    id: MONDO:0012345
    name: arginase 2 deficiency
    synonym: "ARG2" EXACT []

    [Term]
    id: MONDO:0099999
    name: obsolete thing
    synonym: "ITPR1" EXACT []
    is_obsolete: true

    [Term]
    id: http://identifiers.org/hgnc/8140
    name: OPA1

    [Typedef]
    id: has_material_basis_in_germline_mutation_in
    name: has material basis in germline mutation in
    """)

HGNC = textwrap.dedent("""\
    hgnc_id\tsymbol\tname\talias_symbol\tprev_symbol\talias_name\tprev_name\tentrez_id
    HGNC:663\tARG1\targinase 1\tARGX\t\thyperargininemia\t\t383
    HGNC:664\tARG2\targinase 2\tARGX\t\t\t\t384
    HGNC:11957\tMED12\tmediator complex subunit 12\tKIAA0192\tOPA1|HOPA\t\t\t9968
    HGNC:8140\tOPA1\tOPA1 mitochondrial dynamin like GTPase\t\t\t\t\t4976
    HGNC:9791\tRAI1\tretinoic acid induced 1\t\tSMCR\t\tSmith-Magenis syndrome chromosome region\t10743
    HGNC:11123\tSMS\tspermine synthase\t\t\t\t\t6611
    HGNC:6990\tMECP2\tmethyl-CpG binding protein 2\tRTT\t\t\tRett syndrome\t4204
    """)

G2P = textwrap.dedent("""\
    g2p id,gene symbol,hgnc id
    G2P00001,ARG1,663
    G2P00002,ARG2,664
    G2P00747,MED12,11957
    G2P00752,RAI1,HGNC:9791
    G2P00787,SMS,11123
    G2P00600,MECP2,6990
    G2P00601,MECP2,6990
    G2P00999,ZZZ9,99999
    """)

MONDO_WITH_RETT = MONDO_OBO + textwrap.dedent("""\

    [Term]
    id: MONDO:0010726
    name: Rett syndrome
    synonym: "RTT" EXACT []
    is_a: MONDO:0000001
    relationship: has_material_basis_in_germline_mutation_in http://identifiers.org/hgnc/6990 ! MECP2

    [Term]
    id: MONDO:0008814
    name: hyperargininemia
    relationship: has_material_basis_in_germline_mutation_in http://identifiers.org/hgnc/663 ! ARG1
    """)


@pytest.fixture
def lexicon(tmp_path: Path) -> DiseaseLexicon:
    (tmp_path / "mondo.obo").write_text(MONDO_OBO)
    return DiseaseLexicon.from_obo(str(tmp_path / "mondo.obo"))


@pytest.fixture
def built(tmp_path: Path) -> tuple[DiseaseLexicon, HgncPanel]:
    (tmp_path / "mondo.obo").write_text(MONDO_WITH_RETT)
    (tmp_path / "hgnc.txt").write_text(HGNC)
    (tmp_path / "g2p.csv").write_text(G2P)
    lex = DiseaseLexicon.from_obo(str(tmp_path / "mondo.obo"))
    return lex, HgncPanel.from_files(str(tmp_path / "hgnc.txt"), str(tmp_path / "g2p.csv"), lex)


@pytest.fixture
def panel(built) -> HgncPanel:
    return built[1]


def _padded(text: str) -> str:
    return f" {normalise_phrase(text)} "


def test_from_obo_keeps_non_obsolete_mondo_terms_only(lexicon):
    """Labels cover every non-obsolete MONDO: term; obsolete terms and gene terms are absent."""
    assert set(lexicon.labels) == {"MONDO:0000001", "MONDO:0008434", "MONDO:0010679", "MONDO:0012345"}
    assert lexicon.labels["MONDO:0008434"] == "Smith-Magenis syndrome"
    assert not lexicon.is_disease_symbol("ITPR1")
    assert not lexicon.is_disease_symbol("OPA1")


def test_exact_synonyms_are_disease_strings_and_related_synonyms_are_not(lexicon):
    """The exact-string, normalised and abbreviation lookups index labels and EXACT synonyms."""
    assert lexicon.is_disease_symbol("SMS")
    assert lexicon.is_disease_symbol("Smith Magenis syndrome")
    assert not lexicon.is_disease_symbol("SMS syndrome")
    assert lexicon.is_disease_name("smith-magenis SYNDROME")
    assert lexicon.abbrev_upper["SMS"] == {"MONDO:0008434"}
    assert lexicon.abbrev_upper["DMD"] == {"MONDO:0010679"}
    assert "SMITH MAGENIS SYNDROME" not in lexicon.abbrev_upper


def test_label_names_include_the_label_without_a_trailing_syndrome(lexicon):
    assert lexicon.label_names["MONDO:0008434"] == {"smith magenis syndrome", "smith magenis"}
    assert lexicon.label_names["MONDO:0010679"] == {"duchenne muscular dystrophy"}


def test_germline_lineage_is_basis_terms_plus_their_ancestors(lexicon):
    """The sub-lexicon holds terms with a germline-basis gene (relationship or intersection_of)
    and their is_a ancestors, without re-reading the file."""
    ctx = lexicon.germline_lineage()
    assert set(ctx.labels) == {"MONDO:0000001", "MONDO:0008434", "MONDO:0010679"}
    assert "ARG2" not in ctx.abbrev_upper                      # no basis gene, not in the lineage
    assert lexicon.basis_genes["MONDO:0010679"] == {"HGNC:2928"}
    assert ctx.basis_genes is lexicon.basis_genes


def test_disease_context_requires_the_label_and_no_gene_use(lexicon):
    """An abbreviation is a disease where its label (or trimmed label) is written and no
    occurrence is used as a gene."""
    text = "Smith-Magenis syndrome (SMS): a review."
    assert lexicon.disease_context("SMS", text, _padded(text)) == ["MONDO:0008434"]
    text = "Smith-Magenis (SMS) patients."
    assert lexicon.disease_context("SMS", text, _padded(text)) == ["MONDO:0008434"]
    text = "Spermine synthase (SMS) deficiency."
    assert lexicon.disease_context("SMS", text, _padded(text)) == []
    for text in ("Duchenne muscular dystrophy (DMD): DMD gene deletions.",
                 "A missense mutation in DMD causes Duchenne muscular dystrophy (DMD).",
                 "Duchenne muscular dystrophy (DMD); DMD c.1234A>G."):
        assert lexicon.disease_context("DMD", text, _padded(text)) == []
    assert lexicon.disease_context("XYZ", "XYZ syndrome", _padded("XYZ syndrome")) == []


def test_is_disease_here_treats_multi_word_strings_by_name_and_tokens_by_context(lexicon):
    text = "Girls with SMS have regression."
    assert lexicon.is_disease_here("Smith-Magenis syndrome", text, _padded(text))
    assert not lexicon.is_disease_here("SMS", text, _padded(text))
    text = "Smith-Magenis syndrome (SMS)."
    assert lexicon.is_disease_here("SMS", text, _padded(text))


def test_panel_entries_are_keyed_by_hgnc_id(panel):
    """G2P rows join on the hgnc id column with or without the HGNC: prefix; a gene with two
    entries lists both; a row whose HGNC id is unknown is reported, not silently dropped."""
    assert panel.entries["HGNC:9791"] == ["G2P00752"]
    assert panel.entries["HGNC:6990"] == ["G2P00600", "G2P00601"]
    assert panel.symbol_of["HGNC:11957"] == "MED12"
    assert panel.entrez_to_hgnc["4976"] == "HGNC:8140"
    assert "HGNC:8140" not in panel.entries
    assert panel.unresolved_g2p == [("G2P00999", "99999")]


def test_verbatim_dictionary_exclusions(panel):
    """Approved symbols are always in the verbatim dictionary. A previous or alias symbol is
    excluded when it is another HGNC gene's approved symbol or a MONDO disease string; an alias
    shared by two panel genes admits both."""
    assert panel.verbatim["MED12"] == {"HGNC:11957"}
    assert "OPA1" not in panel.verbatim
    assert panel.verbatim["HOPA"] == {"HGNC:11957"}
    assert "RTT" not in panel.verbatim
    assert panel.verbatim["ARGX"] == {"HGNC:663", "HGNC:664"}
    assert panel.verbatim["SMS"] == {"HGNC:11123"}
    by_symbol = {(a["symbol"], a["source"]): a for a in panel.symbol_audit}
    assert by_symbol[("OPA1", "hgnc_previous_symbol")]["reasons"] == "approved_symbol_of_other_gene"
    assert by_symbol[("RTT", "hgnc_alias_symbol")]["reasons"] == "disease_name"
    assert by_symbol[("RTT", "hgnc_alias_symbol")]["disease_caused_by_this_gene"] is True
    assert by_symbol[("SMS", "approved_symbol")]["kept"] is True
    assert by_symbol[("SMS", "approved_symbol")]["disease_basis_genes"] == "RAI1"


def test_name_audit_removes_disease_names_from_the_name_dictionary(panel):
    """HGNC alias and previous names that are MONDO disease names are unindexed and audited."""
    assert {(a["gene"], a["name"], a["source"]) for a in panel.name_audit} == {
        ("ARG1", "hyperargininemia", "hgnc_alias_name"),
        ("MECP2", "Rett syndrome", "hgnc_previous_name"),
    }
    assert panel.matcher.find("Rett syndrome in girls") == set()
    assert panel.matcher.find("methyl-CpG binding protein 2 variants") == {"MECP2"}
    assert panel.matcher.find("hyperargininemia with arginase 1 deficiency") == {"ARG1"}


def test_resolve_candidates_sources_and_report(built):
    """Sources follow the route that found the gene, symbol_match taking precedence, and the
    report carries the audit summary and per-row counts."""
    lex, panel = built
    ctx = lex.germline_lineage()
    rows = [("0", "Fibrillin 1 in SMS variants: spermine synthase."),
            ("1", "Pathogenic ARGX variants."),
            ("2", "MECP2 duplication syndrome in boys."),
            ("3", None)]
    pubtator = {"0": {"6611": {"SMS"}}, "2": {"4204": {"methyl-CpG"}}}
    cands, sources, report = resolve_candidates(rows, pubtator, panel, lex, ctx)
    assert cands == [["G2P00787"], ["G2P00001", "G2P00002"], ["G2P00600", "G2P00601"], []]
    assert sources == [["symbol_match"], ["symbol_fallback", "symbol_fallback"],
                       ["symbol_match", "symbol_match"], []]
    assert report["rows_symbol"] == 2 and report["rows_verbatim"] == 1 and report["rows_name"] == 0
    assert report["pubtator_gene_admitted_by_official_symbol_only"] == 1
    assert report["unresolved_g2p"] == [("G2P00999", "99999")]
    assert {r["source"] for r in report["symbol_audit"]} == {
        "approved_symbol", "hgnc_previous_symbol", "hgnc_alias_symbol"}
