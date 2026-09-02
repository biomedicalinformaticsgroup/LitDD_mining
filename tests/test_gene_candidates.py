"""End-to-end test for the gene-candidate gate (`litdd/pipeline/gene_candidates.py`)."""
from __future__ import annotations

import gzip
import subprocess
import sys
import textwrap
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]

G2P = textwrap.dedent("""\
    g2p id,gene symbol,gene mim,hgnc id,previous gene symbols,disease name,disease mim,disease MONDO,allelic requirement,cross cutting modifier,confidence,variant consequence,variant types,molecular mechanism,molecular mechanism support
    G2P00001,ARG1,111,663,,ARG1-related hyperargininemia,207800,,biallelic_autosomal,,definitive,absent gene product,,loss of function,inferred
    G2P00002,ARG2,222,664,,ARG2-related disorder,,,biallelic_autosomal,,limited,absent gene product,,loss of function,inferred
    G2P00003,FBN1,333,3603,FBN,FBN1-related Marfan syndrome,154700,,monoallelic_autosomal,,definitive,altered gene product structure,missense_variant,dominant negative,evidence
    """)

HGNC = textwrap.dedent("""\
    hgnc_id\tsymbol\tname\talias_name\tprev_name
    HGNC:663\tARG1\targinase 1\t\t
    HGNC:664\tARG2\targinase 2\t\t
    HGNC:3603\tFBN1\tfibrillin 1\t\t
    """)


def _setup(tmp_path):
    (tmp_path / "g2p.csv").write_text(G2P)
    (tmp_path / "hgnc.txt").write_text(HGNC)
    with gzip.open(tmp_path / "gene_info.gz", "wt") as f:
        f.write("#tax_id\tGeneID\tSymbol\n9606\t383\tARG1\n9606\t2200\tFBN1\n")
    with gzip.open(tmp_path / "g2pub.gz", "wt") as f:
        f.write("100\tGene\t383\tARG1\tPubTator3\n")     # symbol match -> ARG1
        f.write("300\tGene\t2200\tFBN1|fibrillin 1\tPubTator3\n")  # mention occurs in TIAB -> FBN1
    pl.DataFrame({
        "pmid": ["100", "200", "300", "400"],
        "tiab": [
            "A homozygous ARG1 variant in hyperargininemia.",      # PubTator
            "The two human arginase genes in hyperargininemia.",   # name only, family -> both
            "Fibrillin 1 in Marfan syndrome.",                     # PubTator + name
            "An abstract about nothing relevant whatsoever.",      # no gene -> dropped
        ],
    }).write_parquet(tmp_path / "in.parquet")
    return tmp_path


def _run(tmp_path, *extra):
    out = tmp_path / "out.parquet"
    cmd = [sys.executable, str(ROOT / "litdd/pipeline/gene_candidates.py"),
           "--input_parquet", str(tmp_path / "in.parquet"),
           "--g2p_csv", str(tmp_path / "g2p.csv"),
           "--gene2pubtator", str(tmp_path / "g2pub.gz"),
           "--gene_info", str(tmp_path / "gene_info.gz"),
           "--out_parquet", str(out), *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return pl.read_parquet(out)


def test_pubtator_loader_separates_text_annotations_from_database_links(tmp_path):
    import gzip

    from litdd.genes import load_pubtator_genes, mention_in_text
    g2p = tmp_path / "gene2pubtator3.gz"
    with gzip.open(g2p, "wt") as f:
        f.write("100\tGene\t1756\tDMD|dystrophin\tPubTator3|gene2pubmed\n")   # text annotation
        f.write("100\tGene\t4204\t\tBioGRID|gene2pubmed\n")                    # database link only
        f.write("100\tGene\t2261\tFGFR3\tPubTator3\n")                         # full-text mention
    gene_info = {"1756": "DMD", "4204": "MECP2", "2261": "FGFR3"}
    legacy = load_pubtator_genes(str(g2p), {"100"}, gene_info)
    assert legacy["100"] == {"DMD", "MECP2", "FGFR3"}                          # old behaviour
    text_only = load_pubtator_genes(str(g2p), {"100"}, gene_info,
                                    text_annotations_only=True, with_mentions=True)
    assert set(text_only["100"]) == {"DMD", "FGFR3"}                           # BioGRID row gone
    tiab = "Dystrophin analysis in muscular dystrophy."
    assert mention_in_text(tiab, text_only["100"]["DMD"], "DMD")               # via mention name
    assert not mention_in_text(tiab, text_only["100"]["FGFR3"], "FGFR3")       # full-text-only gene
    assert mention_in_text("The P-gp transporter", {"P-gp"}, "ABCB1")          # hyphen boundaries


def test_gate_drops_rows_with_no_detected_gene(tmp_path):
    df = _run(_setup(tmp_path))
    assert set(df["pmid"].to_list()) == {"100", "300"}  # 200 needs --hgnc, 400 has no gene


def test_name_matching_recovers_the_protein_name_case(tmp_path):
    tmp_path = _setup(tmp_path)
    df = _run(tmp_path, "--hgnc", str(tmp_path / "hgnc.txt"), "--family_stems")
    rows = {p: (c, s) for p, c, s in zip(df["pmid"], df["candidate_g2p_ids"],
                                         df["candidate_sources"])}
    # PMID 200 says "arginase" and never the symbol: PubTator misses it, and only the
    # enzyme-family stem can see it, so this needs --family_stems. Without that flag the
    # row is dropped -- the precision/recall trade-off is explicit, see the test below.
    assert "200" in rows
    assert set(rows["200"][0]) == {"G2P00001", "G2P00002"}
    assert set(rows["200"][1]) == {"name_match"}


def test_full_name_matching_is_the_default(tmp_path):
    """Without --family_stems the descriptive-family case is not rescued.

    Recorded so the default's cost is visible: full-name matching is precise (a syndrome
    mention can never pull in its whole gene family) but cannot see "arginase" without a
    numeral.
    """
    tmp_path = _setup(tmp_path)
    df = _run(tmp_path, "--hgnc", str(tmp_path / "hgnc.txt"))
    assert "200" not in set(df["pmid"].to_list())


def test_provenance_prefers_symbol_match(tmp_path):
    tmp_path = _setup(tmp_path)
    df = _run(tmp_path, "--hgnc", str(tmp_path / "hgnc.txt"), "--family_stems")
    row = df.filter(pl.col("pmid") == "300")
    assert row["candidate_g2p_ids"].to_list()[0] == ["G2P00003"]
    assert row["candidate_sources"].to_list()[0] == ["symbol_match"]


def test_keep_unmatched_falls_back_to_full_panel(tmp_path):
    tmp_path = _setup(tmp_path)
    df = _run(tmp_path, "--keep_unmatched")
    row = df.filter(pl.col("pmid") == "400")
    assert row.height == 1
    assert len(row["candidate_g2p_ids"].to_list()[0]) == 3   # whole panel
    assert set(row["candidate_sources"].to_list()[0]) == {"fallback_full_panel"}


def test_candidate_mode_emits_the_downstream_contract(tmp_path, monkeypatch):
    """The gene-gated cross-encoder path must emit `top5_cross` in the layout
    `llm_map.py` and `final_data_clean.py` already expect: list<struct<label, score>>.

    The model itself is stubbed -- this is about the data contract, not the scoring.
    """
    import types

    sys.path.insert(0, str(ROOT))
    import litdd.pipeline.crossencode as ce

    tmp_path = _setup(tmp_path)
    cand = _run(tmp_path, "--hgnc", str(tmp_path / "hgnc.txt"))
    cand_path = tmp_path / "cand.parquet"
    cand.write_parquet(cand_path)

    class _Stub:
        def predict(self, pairs):
            return [0.95] * len(pairs)

    monkeypatch.setattr(ce, "load_crossencoder", lambda *a, **k: (_Stub(), "cpu"))
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))

    ce.process_shard_candidates(
        candidates_parquet=str(cand_path),
        g2p_csv=str(tmp_path / "g2p.csv"),
        out_dir=str(tmp_path / "out"),
        skip_if_exists=False,
    )
    out = list((tmp_path / "out").glob("*.parquet"))
    assert len(out) == 1
    df = pl.read_parquet(out[0])
    assert "top5_cross" in df.columns
    first = df["top5_cross"].to_list()[0]
    assert isinstance(first, list) and first, "expected a non-empty candidate list"
    assert set(first[0]) == {"label", "score"}
    assert first[0]["label"].startswith("G2P")   # the full LGMDE thread string
    assert 0.0 <= first[0]["score"] <= 1.0
    # data-driven k: nothing is truncated to 5 by default
    assert all(len(r) == len(c) for r, c in zip(df["top5_cross"].to_list(),
                                                df["candidate_g2p_ids"].to_list()))


def test_symbol_fallback_only_for_pubtator_unannotated_abstracts(tmp_path):
    """PubTator3 has no annotation at all for some (old, title-only) records although the
    gene symbol is verbatim in the text -- 9 curated test abstracts (DMD x6, ITPR1, NEXMIF,
    SCN8A, CPLANE1). --symbol_fallback matches panel symbols verbatim there, and ONLY there:
    an abstract PubTator did annotate must not gain fallback candidates."""
    (tmp_path / "g2p.csv").write_text(G2P + "G2P00004,DMD,444,2928,,DMD-related Duchenne muscular dystrophy,310200,,monoallelic_X_hemizygous,,definitive,absent gene product,,loss of function,inferred\nG2P00005,ITPR1,555,6180,,ITPR1-related spinocerebellar ataxia,,,monoallelic_autosomal,,definitive,altered gene product structure,,dominant negative,inferred\n")
    (tmp_path / "hgnc.txt").write_text(HGNC)
    with gzip.open(tmp_path / "gene_info.gz", "wt") as f:
        f.write("#tax_id\tGeneID\tSymbol\n9606\t383\tARG1\n9606\t1756\tDMD\n")
    with gzip.open(tmp_path / "g2pub.gz", "wt") as f:
        f.write("100\tGene\t383\tARG1\tPubTator3\n")   # annotated: fallback must not apply
    pl.DataFrame({
        "pmid": ["100", "500", "600"],
        "tiab": [
            "A homozygous ARG1 variant; DMD is mentioned in passing.",   # PubTator: ARG1 only
            "DMD carrier detection; long-term follow-up of ITPR1-related disorder.",  # unannotated
            "The CAT scan was normal; SET the MAX dose.",                # blocklisted words only
        ],
    }).write_parquet(tmp_path / "in.parquet")
    without = _run(tmp_path)
    assert without["pmid"].to_list() == ["100"]
    with_fb = _run(tmp_path, "--symbol_fallback")
    rows = {r["pmid"]: r for r in with_fb.to_dicts()}
    assert set(rows) == {"100", "500"}                       # 600 still dropped
    assert rows["100"]["candidate_g2p_ids"] == ["G2P00001"]  # no DMD added to an annotated row
    assert rows["500"]["candidate_g2p_ids"] == ["G2P00004", "G2P00005"]   # "ITPR1-related" -> ITPR1
    assert rows["500"]["candidate_sources"] == ["symbol_fallback", "symbol_fallback"]
