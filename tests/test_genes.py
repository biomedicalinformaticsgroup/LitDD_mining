"""Tests for gene mention detection (`litdd/genes.py`)."""
from __future__ import annotations

import gzip
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from litdd.genes import GeneNameMatcher, load_gene_info, load_pubtator_genes  # noqa: E402

MATCHER = GeneNameMatcher(
    name_to_symbols={"arginase 1": {"ARG1"}, "arginase 2": {"ARG2"}, "fibrillin 1": {"FBN1"}},
    family_to_symbols={"arginase": {"ARG1", "ARG2"}},
)


def test_matches_protein_name_when_symbol_absent():
    """The case this complement exists for (real miss, PMID 2913054)."""
    tiab = ("Differential expression of the two human arginase genes in hyperargininemia. "
            "AI was totally absent in the patient's tissues.")
    assert MATCHER.find(tiab) == {"ARG1", "ARG2"}


def test_longest_match_wins():
    """A specific name must not drag in its siblings via the family stem."""
    assert MATCHER.find("Mutation in arginase 1 causes hyperargininemia.") == {"ARG1"}
    assert MATCHER.find("Both arginase 1 and arginase 2 were assayed.") == {"ARG1", "ARG2"}


def test_family_stem_expands_rather_than_guesses():
    """An unqualified family mention yields every member for downstream disambiguation."""
    assert MATCHER.find("arginase activity was reduced") == {"ARG1", "ARG2"}


def test_no_spurious_matches():
    assert MATCHER.find("No gene names here at all.") == set()
    assert MATCHER.find("") == set()


def test_matching_is_case_and_punctuation_insensitive():
    assert MATCHER.find("FIBRILLIN-1 variants") == {"FBN1"}
    assert MATCHER.find("Fibrillin 1, a matrix protein") == {"FBN1"}


def test_hgnc_loader_restricts_to_whitelist(tmp_path):
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
    # restricted out of the dictionary entirely
    assert m.find("some other protein was measured") == set()
    # previous name is indexed too
    assert m.find("arginase, liver was assayed") == {"ARG1"}


def test_gene_info_and_pubtator_roundtrip(tmp_path):
    gi = tmp_path / "gene_info.gz"
    with gzip.open(gi, "wt") as f:
        f.write("#tax_id\tGeneID\tSymbol\n")
        f.write("9606\t383\tARG1\n")
        f.write("9606\t384\tARG2\n")
        f.write("10090\t11846\tArg1\n")  # mouse: must be excluded
    info = load_gene_info(str(gi))
    assert info == {"383": "ARG1", "384": "ARG2"}

    p2g = tmp_path / "gene2pubtator3.gz"
    with gzip.open(p2g, "wt") as f:
        f.write("2913054\tGene\t383;384\targinase\tPubTator3\n")
        f.write("2913054\tGene\t11846\tArg1\tPubTator3\n")   # mouse id, unmapped
        f.write("9999999\tGene\t383\tARG1\tPubTator3\n")     # other pmid
    genes = load_pubtator_genes(str(p2g), {"2913054"}, info)
    # BOTH ids in the multi-id cell are kept, not just the first
    assert genes == {"2913054": {"ARG1", "ARG2"}}
