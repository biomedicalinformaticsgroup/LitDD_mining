"""End-to-end tests for the cleaning stage (`litdd/pipeline/final_data_clean.py`).

The fixture is built inline: an adjudication parquet, the candidates parquet of the gene gate,
a four-entry G2P panel CSV, a gene2pubtator3 file and a gene_info file.
"""
from __future__ import annotations

import csv
import gzip
import re
import resource
import subprocess
import sys
import textwrap
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = "litdd.pipeline.final_data_clean"

G2P = textwrap.dedent("""\
    g2p id,gene symbol,gene mim,hgnc id,previous gene symbols,disease name,disease mim,disease MONDO,allelic requirement,cross cutting modifier,confidence,variant consequence,variant types,molecular mechanism,molecular mechanism support
    G2P00001,ARG1,111,663,,ARG1-related hyperargininemia,207800,,biallelic_autosomal,,definitive,absent gene product,,loss of function,inferred
    G2P00002,ARG2,222,664,,ARG2-related disorder,,,biallelic_autosomal,,limited,absent gene product,,loss of function,inferred
    G2P00003,FBN1,333,3603,FBN,FBN1-related Marfan syndrome,154700,,monoallelic_autosomal,,definitive,altered gene product structure,missense_variant,dominant negative,evidence
    G2P00004,MECP2,444,6990,,MECP2-related Rett syndrome,312750,,monoallelic_X,,definitive,absent gene product,,loss of function,evidence
    """)

# (pmid, llm_dis_map, candidates offered by the gate)
ROWS = [
    ("100", "G2P00001", ["G2P00001"]),                       # valid and offered
    ("200", "NO MATCH", ["G2P00001", "G2P00003"]),           # NO MATCH
    ("300", "G2P00003", ["G2P00003"]),                       # valid and offered
    ("400", "G2P99999", ["G2P00001"]),                       # id not in the panel
    ("500", None, ["G2P00004"]),                             # null answer
    ("600", "G2P00001;G2P00003", ["G2P00001", "G2P00003"]),  # two ids, both offered
    ("700", "G2P00002", ["G2P00001"]),                       # in the panel, not offered
    ("800", "'G2P00004'", ["G2P00004"]),                     # quoted id
    ("900", "G2P00001;G2P00002", ["G2P00001"]),              # one offered, one not
    ("007", "G2P00004", ["G2P00004"]),                       # leading zero
    ("1000", "", ["G2P00002"]),                              # empty answer
    ("1100", "G2P00001", None),                              # no candidates row
]


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("clean")
    (d / "g2p.csv").write_text(G2P)
    with gzip.open(d / "gene_info.gz", "wt") as f:
        f.write("#tax_id\tGeneID\tSymbol\tLocusTag\n")
        f.write("9606\t383\tARG1\t-\n9606\t384\tARG2\t-\n9606\t2200\tFBN1\t-\n9606\t4204\tMECP2\t-\n")
        f.write("10090\t11846\tArg1\t-\n")                       # mouse: not human
    with gzip.open(d / "gene2pubtator3.gz", "wt") as f:
        f.write("100\tGene\t383\tARG1\tPubTator3\n")
        f.write("200\tGene\t383;2200\tARG1|FBN1\tPubTator3\n")   # two GeneIDs in one cell
        f.write("500\tGene\t4204\tMECP2\tPubTator3\n")
        f.write("1000\tGene\t11846\tArg1\tPubTator3\n")           # mouse gene: dropped
    llm = pa.table({
        "pmid": pa.array([r[0] for r in ROWS], pa.string()),
        "tiab": pa.array([f"abstract {r[0]}" for r in ROWS], pa.string()),
        "llm_dis_map": pa.array([r[1] for r in ROWS], pa.string()),
        "candidates": pa.array([r[2] for r in ROWS], pa.list_(pa.string())),
    })
    pq.write_table(llm, d / "llm.parquet", row_group_size=4)
    pq.write_table(llm.drop_columns(["candidates"]), d / "llm_no_candidates.parquet", row_group_size=4)
    with_rows = [r for r in ROWS if r[2] is not None]
    pq.write_table(pa.table({
        "pmid": pa.array([r[0] for r in with_rows], pa.string()),
        "candidate_g2p_ids": pa.array([r[2] for r in with_rows], pa.list_(pa.string())),
    }), d / "candidates.parquet")
    return d


def base_cmd(d: Path, out: Path, llm: str = "llm.parquet") -> list[str]:
    return [
        sys.executable, "-m", MODULE,
        "--llm_file", str(d / llm),
        "--g2p_file", str(d / "g2p.csv"),
        "--candidates_parquet", str(d / "candidates.parquet"),
        "--output_csv", str(out),
    ]


def run_cleaner(d: Path, out: Path, *extra: str, llm: str = "llm.parquet") -> subprocess.CompletedProcess:
    r = subprocess.run(base_cmd(d, out, llm) + list(extra), capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    return r


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def pairs(path: Path) -> set[tuple[str, str]]:
    return {(r["PMID"], r["G2P_IDs"]) for r in read_csv(path)}


def test_help_exits_zero_and_lists_the_flags():
    """`--help` exits 0 and names every flag of the cleaning stage."""
    r = subprocess.run([sys.executable, "-m", MODULE, "--help"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0
    for flag in ("--llm_file", "--g2p_file", "--candidates_parquet", "--gene2pubtator",
                 "--gene_info", "--output_csv", "--no_match_csv", "--debug"):
        assert flag in r.stdout
    # No other long option is offered (options start their own line in the help text).
    offered = set(re.findall(r"^\s+(?:-h, )?(--[a-z0-9_]+)", r.stdout, flags=re.M))
    assert offered == {"--help", "--llm_file", "--g2p_file", "--candidates_parquet", "--gene2pubtator",
                       "--gene_info", "--output_csv", "--no_match_csv", "--debug"}


def test_candidates_parquet_is_required(fixture_dir, tmp_path):
    """Omitting --candidates_parquet is a usage error."""
    cmd = [sys.executable, "-m", MODULE, "--llm_file", str(fixture_dir / "llm.parquet"),
           "--g2p_file", str(fixture_dir / "g2p.csv"), "--output_csv", str(tmp_path / "o.csv")]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0
    assert "--candidates_parquet" in r.stderr


def test_output_schema(fixture_dir, tmp_path):
    """The output has the header PMID,G2P_IDs and no empty cells."""
    out = tmp_path / "out.csv"
    run_cleaner(fixture_dir, out)
    with open(out) as f:
        assert f.readline().strip() == "PMID,G2P_IDs"
    rows = read_csv(out)
    assert rows
    assert all(r["PMID"] and r["G2P_IDs"] for r in rows)


def test_ids_outside_the_panel_are_dropped(fixture_dir, tmp_path):
    """An answer naming an id absent from the G2P CSV produces no row."""
    out = tmp_path / "out.csv"
    run_cleaner(fixture_dir, out)
    kept = pairs(out)
    assert not any(g == "G2P99999" for _, g in kept)
    assert not any(p == "400" for p, _ in kept)


def test_ids_not_offered_for_the_abstract_are_dropped(fixture_dir, tmp_path):
    """A panel id that the gate did not offer for that abstract produces no row."""
    out = tmp_path / "out.csv"
    run_cleaner(fixture_dir, out)
    kept = pairs(out)
    assert ("700", "G2P00002") not in kept          # in the panel, offered elsewhere only
    assert ("1100", "G2P00001") not in kept         # abstract has no candidate row at all
    assert ("900", "G2P00001") in kept and ("900", "G2P00002") not in kept


def test_two_ids_both_offered_are_both_kept(fixture_dir, tmp_path):
    """A `;`-separated answer yields one row per id when every id was offered."""
    out = tmp_path / "out.csv"
    run_cleaner(fixture_dir, out)
    kept = pairs(out)
    assert ("600", "G2P00001") in kept and ("600", "G2P00003") in kept


def test_quoted_ids_are_stripped(fixture_dir, tmp_path):
    """Quote characters around an id are removed before the checks."""
    out = tmp_path / "out.csv"
    run_cleaner(fixture_dir, out)
    assert ("800", "G2P00004") in pairs(out)


def test_expected_final_table(fixture_dir, tmp_path):
    """The accepted set is exactly the valid, offered ids of the fixture, in input order."""
    out = tmp_path / "out.csv"
    run_cleaner(fixture_dir, out)
    assert [(r["PMID"], r["G2P_IDs"]) for r in read_csv(out)] == [
        ("100", "G2P00001"), ("300", "G2P00003"), ("600", "G2P00001"), ("600", "G2P00003"),
        ("800", "G2P00004"), ("900", "G2P00001"), ("007", "G2P00004"),
    ]


def test_no_match_rows_with_genes_mentioned(fixture_dir, tmp_path):
    """NO MATCH, null and empty answers go to the no-match CSV with gene mentions and counts.

    `genes_mentioned` includes every GeneID of a multi-id PubTator cell, resolved to human
    symbols; `n_candidates` is the length of the row's candidate list.
    """
    out, nomatch = tmp_path / "out.csv", tmp_path / "nomatch.csv"
    run_cleaner(fixture_dir, out, "--no_match_csv", str(nomatch),
                "--gene2pubtator", str(fixture_dir / "gene2pubtator3.gz"),
                "--gene_info", str(fixture_dir / "gene_info.gz"))
    with open(nomatch) as f:
        assert f.readline().strip() == "pmid,genes_mentioned,n_candidates"
    rows = {r["pmid"]: r for r in read_csv(nomatch)}
    assert set(rows) == {"200", "500", "1000"}
    assert rows["200"]["genes_mentioned"] == "ARG1;FBN1"
    assert rows["500"]["genes_mentioned"] == "MECP2"
    assert rows["1000"]["genes_mentioned"] == ""
    assert {p: r["n_candidates"] for p, r in rows.items()} == {"200": "2", "500": "1", "1000": "1"}


def test_no_match_without_gene_resources(fixture_dir, tmp_path):
    """Without --gene2pubtator and --gene_info the no-match file is written with empty gene mentions."""
    out, nomatch = tmp_path / "out.csv", tmp_path / "nomatch.csv"
    run_cleaner(fixture_dir, out, "--no_match_csv", str(nomatch))
    rows = read_csv(nomatch)
    assert {r["pmid"] for r in rows} == {"200", "500", "1000"}
    assert all(r["genes_mentioned"] == "" for r in rows)


def test_n_candidates_falls_back_to_the_candidates_parquet(fixture_dir, tmp_path):
    """When the LLM parquet has no `candidates` column, `n_candidates` comes from candidates.parquet."""
    out, nomatch = tmp_path / "out.csv", tmp_path / "nomatch.csv"
    run_cleaner(fixture_dir, out, "--no_match_csv", str(nomatch), llm="llm_no_candidates.parquet")
    rows = {r["pmid"]: r["n_candidates"] for r in read_csv(nomatch)}
    assert rows == {"200": "2", "500": "1", "1000": "1"}
    assert pairs(out) == pairs(tmp_path / "out.csv")


def test_pmid_typing_preserves_strings(fixture_dir, tmp_path):
    """PMIDs are written as the strings read from the parquet, leading zeros included."""
    out = tmp_path / "out.csv"
    run_cleaner(fixture_dir, out)
    written = {r["PMID"] for r in read_csv(out)}
    assert "007" in written
    parquet_pmids = set(pq.read_table(fixture_dir / "llm.parquet", columns=["pmid"]).column(0).to_pylist())
    assert written <= parquet_pmids


def test_gene_resources_must_be_given_together(fixture_dir, tmp_path):
    """--gene2pubtator without --gene_info (or the reverse) exits non-zero."""
    cmd = base_cmd(fixture_dir, tmp_path / "o.csv") + ["--gene2pubtator", str(fixture_dir / "gene2pubtator3.gz")]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 1
    assert "together" in r.stderr


def test_missing_input_exits_one(fixture_dir, tmp_path):
    """A missing input path is reported and the exit status is 1."""
    cmd = base_cmd(fixture_dir, tmp_path / "o.csv")
    cmd[cmd.index("--llm_file") + 1] = str(fixture_dir / "absent.parquet")
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 1
    assert "not found" in r.stderr


def test_debug_exits_zero(fixture_dir, tmp_path):
    """`--debug` logs each decision and does not change the output."""
    out = tmp_path / "out.csv"
    r = run_cleaner(fixture_dir, out, "--debug")
    assert "NOT_IN_PANEL" in r.stderr and "NOT_OFFERED_AS_CANDIDATE" in r.stderr and "VALID" in r.stderr
    plain = tmp_path / "plain.csv"
    run_cleaner(fixture_dir, plain)
    assert out.read_text() == plain.read_text()


def test_streaming_memory(fixture_dir, tmp_path):
    """The cleaner reads the parquet in batches; peak child RSS stays under 1 GB."""
    out = tmp_path / "out.csv"
    cmd = ["/usr/bin/env", "bash", "-c", "exec " + " ".join(base_cmd(fixture_dir, out))]
    subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=ROOT)
    peak_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss  # KB on Linux
    assert peak_kb < 1_000_000, f"peak child RSS {peak_kb} KB > 1 GB"
