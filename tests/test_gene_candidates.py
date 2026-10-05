"""End-to-end tests for the gene gate CLI (``python -m litdd.pipeline.gene_candidates``)."""
from __future__ import annotations

import gzip
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]

G2P = textwrap.dedent("""\
    g2p id,gene symbol,hgnc id,previous gene symbols,disease name
    G2P00001,ARG1,663,,ARG1-related hyperargininemia
    G2P00002,ARG2,664,,ARG2-related disorder
    G2P00003,FBN1,3603,FBN,FBN1-related Marfan syndrome
    G2P00004,DMD,2928,,DMD-related Duchenne muscular dystrophy
    G2P00005,ITPR1,6180,,ITPR1-related spinocerebellar ataxia
    G2P00747,MED12,11957,OPA1; HOPA,MED12-related disorder
    G2P00752,RAI1,HGNC:9791,SMCR,Smith-Magenis syndrome
    G2P00787,SMS,11123,,Snyder-Robinson syndrome
    G2P00600,MECP2,6990,RTT,Rett syndrome
    """)

HGNC = textwrap.dedent("""\
    hgnc_id\tsymbol\tname\talias_symbol\tprev_symbol\talias_name\tprev_name\tentrez_id
    HGNC:663\tARG1\targinase 1\t\t\t\t\t383
    HGNC:664\tARG2\targinase 2\t\t\t\t\t384
    HGNC:3603\tFBN1\tfibrillin 1\t\tFBN\t\t\t2200
    HGNC:2928\tDMD\tdystrophin\t\t\t\t\t1756
    HGNC:6180\tITPR1\tinositol 1,4,5-trisphosphate receptor type 1\t\t\t\t\t3708
    HGNC:11957\tMED12\tmediator complex subunit 12\tKIAA0192\tOPA1|HOPA\t\t\t9968
    HGNC:8140\tOPA1\tOPA1 mitochondrial dynamin like GTPase\t\t\t\t\t4976
    HGNC:9791\tRAI1\tretinoic acid induced 1\t\tSMCR\t\tSmith-Magenis syndrome chromosome region\t10743
    HGNC:11123\tSMS\tspermine synthase\t\t\t\t\t6611
    HGNC:6990\tMECP2\tmethyl-CpG binding protein 2\tRTT\t\t\tRett syndrome\t4204
    """)

MONDO_OBO = textwrap.dedent("""\
    [Term]
    id: MONDO:0008434
    name: Smith-Magenis syndrome
    synonym: "SMS" EXACT []
    relationship: has_material_basis_in_germline_mutation_in http://identifiers.org/hgnc/9791 ! RAI1

    [Term]
    id: MONDO:0010726
    name: Rett syndrome
    synonym: "RTT" EXACT []
    relationship: has_material_basis_in_germline_mutation_in http://identifiers.org/hgnc/6990 ! MECP2

    [Term]
    id: MONDO:0010679
    name: Duchenne muscular dystrophy
    synonym: "DMD" EXACT []
    relationship: has_material_basis_in_germline_mutation_in http://identifiers.org/hgnc/2928 ! DMD

    [Term]
    id: http://identifiers.org/hgnc/8140
    name: OPA1
    """)


def _run(tmp_path: Path, tiabs: list[str], pubtator_rows: list[tuple[str, ...]] = (),
         audit: bool = False) -> pl.DataFrame:
    """Write the fixtures and run the gate; pmids are "0", "1", ... in ``tiabs`` order."""
    (tmp_path / "g2p.csv").write_text(G2P)
    (tmp_path / "hgnc.txt").write_text(HGNC)
    (tmp_path / "mondo.obo").write_text(MONDO_OBO)
    with gzip.open(tmp_path / "g2pub.gz", "wt") as f:
        for row in pubtator_rows:
            f.write("\t".join(row) + "\n")
    pl.DataFrame({"pmid": [str(i) for i in range(len(tiabs))], "tiab": tiabs}).write_parquet(
        tmp_path / "in.parquet")
    out = tmp_path / "out.parquet"
    cmd = [sys.executable, "-m", "litdd.pipeline.gene_candidates",
           "--input_parquet", str(tmp_path / "in.parquet"),
           "--g2p_csv", str(tmp_path / "g2p.csv"),
           "--gene2pubtator", str(tmp_path / "g2pub.gz"),
           "--hgnc", str(tmp_path / "hgnc.txt"),
           "--mondo_obo", str(tmp_path / "mondo.obo"),
           "--out_parquet", str(out)]
    if audit:
        cmd += ["--audit_prefix", str(tmp_path / "audit" / "gate")]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    return pl.read_parquet(out)


def _rows(out: pl.DataFrame) -> dict[str, dict]:
    return {r["pmid"]: r for r in out.iter_rows(named=True)}


def _pub(pmid: str, gene_id: str, mention: str) -> tuple[str, ...]:
    return (pmid, "Gene", gene_id, mention, "PubTator3")


def test_pubtator_loaders_separate_text_annotations_from_database_links(tmp_path):
    """load_pubtator_gene_ids keeps PubTator3 text annotations with their mention strings;
    load_pubtator_genes returns symbols from every resource."""
    from litdd.genes import load_pubtator_gene_ids, load_pubtator_genes, mention_in_text
    g2p = tmp_path / "gene2pubtator3.gz"
    with gzip.open(g2p, "wt") as f:
        f.write("100\tGene\t1756\tDMD|dystrophin\tPubTator3|gene2pubmed\n")   # text annotation
        f.write("100\tGene\t4204\t\tBioGRID|gene2pubmed\n")                    # database link only
        f.write("100\tGene\t2261;2260\tFGFR3\tPubTator3\n")                    # two GeneIDs in one cell
    gene_info = {"1756": "DMD", "4204": "MECP2", "2261": "FGFR3", "2260": "FGFR1"}
    assert load_pubtator_genes(str(g2p), {"100"}, gene_info) == {"100": {"DMD", "MECP2", "FGFR3", "FGFR1"}}
    ids = load_pubtator_gene_ids(str(g2p), {"100"})
    assert ids == {"100": {"1756": {"DMD", "dystrophin"}, "2261": {"FGFR3"}, "2260": {"FGFR3"}}}
    tiab = "Dystrophin analysis in muscular dystrophy."
    assert mention_in_text(tiab, ids["100"]["1756"], "DMD")
    assert not mention_in_text(tiab, ids["100"]["2261"], "FGFR3")
    assert mention_in_text("The P-gp transporter", {"P-gp"}, "ABCB1")


def test_gate_drops_rows_with_no_detected_gene(tmp_path):
    """A row whose text names no panel gene by any route is not written."""
    out = _run(tmp_path, ["A homozygous ARG1 variant in hyperargininemia.",
                          "An abstract about nothing relevant whatsoever."],
               [_pub("0", "383", "ARG1")])
    assert out["pmid"].to_list() == ["0"]
    assert out.columns[-2:] == ["candidate_g2p_ids", "candidate_sources"]


def test_name_matching_recovers_the_protein_name_case(tmp_path):
    """A gene written only by its HGNC descriptive name is found with source name_match."""
    rows = _rows(_run(tmp_path, ["Fibrillin 1 in Marfan syndrome.",
                                 "Methyl-CpG binding protein 2 dysfunction in girls."]))
    assert rows["0"]["candidate_g2p_ids"] == ["G2P00003"]
    assert rows["0"]["candidate_sources"] == ["name_match"]
    assert rows["1"]["candidate_g2p_ids"] == ["G2P00600"]


def test_provenance_prefers_symbol_match(tmp_path):
    """A gene found by PubTator and by name is recorded once, as symbol_match."""
    rows = _rows(_run(tmp_path, ["Fibrillin 1 in Marfan syndrome."],
                      [_pub("0", "2200", "FBN1|fibrillin 1")]))
    assert rows["0"]["candidate_g2p_ids"] == ["G2P00003"]
    assert rows["0"]["candidate_sources"] == ["symbol_match"]


def test_symbol_fallback_only_where_pubtator_verified_no_panel_gene(tmp_path):
    """Verbatim symbols are matched only in abstracts where PubTator3 verified no panel gene;
    blocklisted symbols never match."""
    tiabs = [
        "A homozygous ARG1 variant; DMD is mentioned in passing.",              # PubTator: ARG1
        "DMD carrier detection; long-term follow-up of ITPR1-related disorder.",  # unannotated
        "The CAT scan was normal; SET the MAX dose.",                           # blocklist only
    ]
    rows = _rows(_run(tmp_path, tiabs, [_pub("0", "383", "ARG1")]))
    assert set(rows) == {"0", "1"}
    assert rows["0"]["candidate_g2p_ids"] == ["G2P00001"]
    assert rows["1"]["candidate_g2p_ids"] == ["G2P00004", "G2P00005"]
    assert rows["1"]["candidate_sources"] == ["symbol_fallback", "symbol_fallback"]


def test_another_genes_previous_symbol_is_not_followed(tmp_path):
    """PubTator's OPA1 (GeneID 4976) does not resolve to MED12 although G2P and HGNC list OPA1
    as a previous symbol of MED12, and the verbatim dictionary excludes OPA1 for the same reason."""
    out = _run(tmp_path, ["OPA1 variants in optic atrophy."], [_pub("0", "4976", "OPA1")])
    assert out.height == 0
    out = _run(tmp_path, ["OPA1 variants in optic atrophy.", "HOPA variants."])
    assert _rows(out).keys() == {"1"}          # HOPA, a previous symbol nobody else owns, is kept


def test_disease_name_aliases_excluded_and_disease_context_applied(tmp_path):
    """RTT (a MONDO synonym) is not in the verbatim dictionary; SMS counts as spermine synthase
    only where Smith-Magenis syndrome is not also written, or where SMS is used as a gene."""
    tiabs = [
        "Smith-Magenis syndrome (SMS): clinical review of 20 patients.",   # SMS is the disease
        "SMS variants cause Snyder-Robinson syndrome.",                   # SMS is the gene
        "Girls with RTT have regression.",                                # verbatim RTT excluded
        "A RAI1 frameshift in Smith-Magenis syndrome (SMS).",            # RAI1 via PubTator, not SMS
        "Girls with Rett syndrome (RTT) have regression.",                # PubTator RTT in context
        "Duchenne muscular dystrophy (DMD): DMD gene deletions.",         # gene use keeps DMD
        "Duchenne muscular dystrophy (DMD) in boys.",                     # DMD is the disease
    ]
    pub = [_pub("0", "6611", "SMS"), _pub("1", "6611", "SMS"), _pub("3", "10743", "RAI1"),
           _pub("3", "6611", "SMS"), _pub("4", "4204", "RTT")]
    rows = _rows(_run(tmp_path, tiabs, pub, audit=True))
    assert set(rows) == {"1", "3", "5"}
    assert rows["1"]["candidate_g2p_ids"] == ["G2P00787"]
    assert rows["3"]["candidate_g2p_ids"] == ["G2P00752"]
    assert rows["5"]["candidate_sources"] == ["symbol_fallback"]

    symbols = pl.read_csv(tmp_path / "audit" / "gate_symbols.tsv", separator="\t")
    rtt = symbols.filter(pl.col("symbol") == "RTT").to_dicts()[0]
    assert rtt["kept"] is False and rtt["reasons"] == "disease_name"
    names = pl.read_csv(tmp_path / "audit" / "gate_names.tsv", separator="\t")
    assert names["name"].to_list() == ["Rett syndrome"]
    stats = json.loads((tmp_path / "audit" / "gate_stats.json").read_text())
    # SMS in row 0 is declined on the PubTator route and again on the verbatim route, SMS in
    # row 3 on the PubTator route, DMD in row 6 on the verbatim route.
    assert stats["disease_context_abstract_symbol_pairs"] == 4
    assert stats["disease_mentions_dropped"] == 3                     # SMS in rows 0 and 3, RTT in row 4
    assert stats["rows_symbol"] == 2 and stats["rows_verbatim"] == 1
