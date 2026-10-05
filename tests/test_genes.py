"""Tests for the gene-data loaders and text-matching primitives (``litdd/genes.py``)."""
from __future__ import annotations

import gzip
import textwrap

from litdd.genes import (
    GeneNameMatcher,
    g2p_hgnc_ids,
    g2p_ids,
    load_gene_info,
    load_pubtator_gene_ids,
    load_pubtator_genes,
    normalise_phrase,
    split_pipe,
    symbol_admissible,
    symbol_written,
    uppercase_tokens,
)

MATCHER = GeneNameMatcher(
    name_to_symbols={"arginase 1": {"ARG1"}, "arginase 2": {"ARG2"}, "fibrillin 1": {"FBN1"}},
)

HGNC_DISEASE_NAMED = textwrap.dedent("""\
    hgnc_id\tsymbol\tname\talias_name\tprev_name
    HGNC:966\tBBS1\tBardet-Biedl syndrome 1\t\t
    HGNC:967\tBBS2\tBardet-Biedl syndrome 2\t\t
    HGNC:968\tBBS4\tBardet-Biedl syndrome 4\t\t
    HGNC:663\tARG1\targinase 1\t\t
    HGNC:664\tARG2\targinase 2\t\t
    """)


def test_normalise_phrase_and_split_pipe():
    """normalise_phrase drops punctuation and case; split_pipe splits HGNC fields on '|' and
    strips quotes and blanks."""
    assert normalise_phrase("Smith-Magenis SYNDROME, type 2") == "smith magenis syndrome type 2"
    assert split_pipe('"OPA1|HOPA"') == ["OPA1", "HOPA"]
    assert split_pipe(" a | | b ") == ["a", "b"]
    assert split_pipe(None) == [] and split_pipe("") == []


def test_g2p_loaders(tmp_path):
    """g2p_ids returns the id set; g2p_hgnc_ids maps each id to the hgnc id column as written."""
    p = tmp_path / "g2p.csv"
    p.write_text("g2p id,gene symbol,hgnc id\nG2P00001,ARG1,663\nG2P00002,RAI1,HGNC:9791\n,X,1\n")
    assert g2p_ids(str(p)) == {"G2P00001", "G2P00002"}
    assert g2p_hgnc_ids(str(p)) == {"G2P00001": "663", "G2P00002": "HGNC:9791"}


def test_disease_named_genes_do_not_match_on_the_disease_alone(tmp_path):
    """A name that is a disease label plus an index matches only when written in full."""
    p = tmp_path / "hgnc_disease.txt"
    p.write_text(HGNC_DISEASE_NAMED)
    m = GeneNameMatcher.from_hgnc(str(p), {"BBS1", "BBS2", "BBS4", "ARG1", "ARG2"})
    assert m.find("A fifth locus for Bardet-Biedl syndrome maps to 2q31.") == set()
    assert m.find("A truncating Bardet-Biedl syndrome 1 variant") == {"BBS1"}
    assert m.find("the two human arginase genes") == set()


def test_matches_protein_name_when_symbol_absent():
    tiab = "Arginase 1 was totally absent in the patient's tissues; the ARG1 symbol is not used here."
    assert MATCHER.find(tiab) == {"ARG1"}


def test_longest_match_wins():
    """Each token is consumed by the longest indexed name covering it."""
    assert MATCHER.find("Mutation in arginase 1 causes hyperargininemia.") == {"ARG1"}
    assert MATCHER.find("Both arginase 1 and arginase 2 were assayed.") == {"ARG1", "ARG2"}


def test_no_spurious_matches():
    assert MATCHER.find("No gene names here at all.") == set()
    assert MATCHER.find("") == set()


def test_matching_is_case_and_punctuation_insensitive():
    assert MATCHER.find("FIBRILLIN-1 variants") == {"FBN1"}
    assert MATCHER.find("Fibrillin 1, a matrix protein") == {"FBN1"}


def test_hgnc_loader_restricts_to_whitelist(tmp_path):
    """Only genes in keep_symbols are indexed; previous names are indexed as well as names."""
    hgnc = tmp_path / "hgnc.txt"
    hgnc.write_text(textwrap.dedent("""\
        hgnc_id\tsymbol\tname\talias_name\tprev_name
        HGNC:663\tARG1\targinase 1\t\targinase, liver
        HGNC:664\tARG2\targinase 2\t\t
        HGNC:3603\tFBN1\tfibrillin 1\t\t
        HGNC:9999\tZZZ9\tsome other protein\t\t
        """))
    m = GeneNameMatcher.from_hgnc(str(hgnc), keep_symbols={"ARG1", "ARG2"})
    assert m.find("arginase 1 deficiency") == {"ARG1"}
    assert m.find("some other protein was measured") == set()
    assert m.find("arginase, liver was assayed") == {"ARG1"}


def test_gene_info_and_pubtator_roundtrip(tmp_path):
    """load_gene_info keeps human rows; load_pubtator_genes keeps every GeneID of a multi-id
    cell and every resource; load_pubtator_gene_ids keeps PubTator3 rows with mention strings."""
    gi = tmp_path / "gene_info.gz"
    with gzip.open(gi, "wt") as f:
        f.write("#tax_id\tGeneID\tSymbol\n")
        f.write("9606\t383\tARG1\n")
        f.write("9606\t384\tARG2\n")
        f.write("10090\t11846\tArg1\n")  # mouse
    info = load_gene_info(str(gi))
    assert info == {"383": "ARG1", "384": "ARG2"}

    p2g = tmp_path / "gene2pubtator3.gz"
    with gzip.open(p2g, "wt") as f:
        f.write("2913054\tGene\t383;384\targinase\tPubTator3\n")
        f.write("2913054\tGene\t11846\tArg1\tPubTator3\n")   # mouse id, unmapped
        f.write("2913054\tGene\t2200\t\tBioGRID\n")           # database link, no mention
        f.write("9999999\tGene\t383\tARG1\tPubTator3\n")     # other pmid
    assert load_pubtator_genes(str(p2g), {"2913054"}, info) == {"2913054": {"ARG1", "ARG2"}}
    assert load_pubtator_genes(str(p2g), {"2913054"}, {**info, "2200": "FBN1"}) == {
        "2913054": {"ARG1", "ARG2", "FBN1"}}
    assert load_pubtator_gene_ids(str(p2g), {"2913054"}) == {
        "2913054": {"383": {"arginase"}, "384": {"arginase"}, "11846": {"Arg1"}}}
    assert set(load_pubtator_gene_ids(str(p2g), None)) == {"2913054", "9999999"}


def test_uppercase_tokens_keep_hyphenated_symbols_whole():
    """NKX2-1 stays one token; a hyphen followed by lower case ends the token."""
    assert uppercase_tokens("NKX2-1 and ITPR1-related disease; HLA-DRB1.") == {"NKX2-1", "ITPR1", "HLA-DRB1"}
    assert uppercase_tokens("lowercase only") == set()


def test_symbol_written_is_case_sensitive_word_bounded_and_blocklisted():
    text = "The CAT scan; ARG1 variants; arg2 lowered; ARG10 differs."
    lowered = text.lower()
    assert symbol_written("ARG1", text, lowered)
    assert not symbol_written("ARG2", text, lowered)        # case-sensitive
    assert not symbol_written("ARG", text, lowered)         # word-bounded
    assert not symbol_written("CAT", text, lowered)         # blocklisted
    assert not symbol_written("IQ", "IQ was 70", "iq was 70")   # shorter than three characters


def test_ambiguous_symbols_are_declined_by_context():
    """A document-level competing expansion, or every occurrence being part of a longer name
    or product name, declines the symbol; one clean occurrence admits it."""
    cad = "Premature coronary artery disease (CAD) in a family."
    assert not symbol_admissible("CAD", cad, cad.lower())
    cad = "Biallelic CAD variants cause epileptic encephalopathy."
    assert symbol_admissible("CAD", cad, cad.lower())
    sets = "The SET domain and SET binding factor 1."
    assert not symbol_admissible("SET", sets, sets.lower())
    sets = "Variants in the SET gene; a SET domain protein."
    assert symbol_admissible("SET", sets, sets.lower())
    kit = "Analysed with the SALSA MLPA KIT P245."
    assert not symbol_admissible("KIT", kit, kit.lower())
    kit = "KIT variants in piebaldism."
    assert symbol_admissible("KIT", kit, kit.lower())


HGNC_COMPOUND = textwrap.dedent("""\
    hgnc_id\tsymbol\tname\talias_name\tprev_name
    HGNC:4556\tGNE\tglucosamine (UDP-N-acetyl)-2-epimerase/N-acetylmannosamine kinase\tbifunctional UDP-N-acetylglucosamine 2-epimerase/N-acetylmannosamine kinase\t
    HGNC:5172\tHR\tHR lysine demethylase and nuclear receptor corepressor\t\thairless homolog (mouse)
    HGNC:4193\tGCH1\tGTP cyclohydrolase 1\tGTP cyclohydrolase I\t
    HGNC:1421\tCDKL5\tcyclin dependent kinase like 5\tserine/threonine kinase 9\t
    """)


def _compound(tmp_path):
    p = tmp_path / "hgnc_compound.txt"
    p.write_text(HGNC_COMPOUND)
    return GeneNameMatcher.from_hgnc(str(p), {"GNE", "HR", "GCH1", "CDKL5"})


def test_compound_names_match_each_slash_part_and_drop_parentheticals(tmp_path):
    """A slash-joined bifunctional enzyme name matches on either part; a parenthetical
    orthologue tag and the word homolog are not required."""
    m = _compound(tmp_path)
    assert m.find("Mutations in the human UDP-N-acetylglucosamine 2-epimerase gene") == {"GNE"}
    assert m.find("Genomic organization of the human hairless gene (HR)") == {"HR"}


def test_roman_and_arabic_index_forms_both_resolve(tmp_path):
    """An alias name with a roman index and the approved name with an arabic one are one gene."""
    m = _compound(tmp_path)
    assert m.find("A new GTP-cyclohydrolase I mutation in dopa-responsive dystonia") == {"GCH1"}


def test_slash_split_does_not_index_single_word_parts(tmp_path):
    """A slash part of one word ("serine") is not indexed; two-word parts are."""
    m = _compound(tmp_path)
    assert m.find("Substitution of glycine-661 by serine in the alpha1(I) chains") == set()
    assert m.find("threonine kinase 9 variants") == {"CDKL5"}
