"""Tests for Diana's `final_data_clean.py` bug-fix.

These tests run on small fixtures under `tests/fixtures/` and exercise both
the new bug-fix (gene-match check applied even with `--score_cutoff 0`) and
the schema/streaming guarantees of the merged v1+v2 cleaner.

Build the fixtures once with `python tests/build_fixtures.py`. Run with
`pytest tests/test_final_data_clean.py -v`.
"""
from __future__ import annotations

import csv
import gzip
import resource
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "annotate_pubmed" / "final_data_clean.py"
FIX = Path(__file__).resolve().parent / "fixtures"
LLM = FIX / "llm_shard_sample.parquet"
G2P = FIX / "g2p_sample.csv"
GENE2PUBTATOR = FIX / "gene2pubtator_sample.tsv.gz"
GENE_INFO = FIX / "gene_info_sample.gz"


def run_cleaner(out_csv: Path, score_cutoff: float, gene_info: bool = True,
                debug: bool = False, no_gene_check: bool = False) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable, str(SCRIPT),
        "--llm_file", str(LLM),
        "--g2p_file", str(G2P),
        "--gene2pubtator", str(GENE2PUBTATOR),
        "--score_cutoff", str(score_cutoff),
        "--output_csv", str(out_csv),
    ]
    if gene_info:
        cmd += ["--gene_info", str(GENE_INFO)]
    if no_gene_check:
        cmd.append("--no_gene_check")
    if debug:
        cmd.append("--debug")
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def read_output(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def fixtures_present() -> None:
    for p in (LLM, G2P, GENE2PUBTATOR, GENE_INFO):
        assert p.exists(), f"fixture missing: {p} — run tests/build_fixtures.py"


def test_help_exits_zero():
    """CLI works (argparse covers required args)."""
    r = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True)
    assert r.returncode == 0
    assert "--llm_file" in r.stdout
    assert "--gene_info" in r.stdout


def test_output_schema(tmp_path, fixtures_present):
    """Output has exactly two columns and no empty rows."""
    out = tmp_path / "out.csv"
    run_cleaner(out, score_cutoff=0.9)
    rows = read_output(out)
    with open(out) as f:
        header = f.readline().strip().split(",")
    assert header == ["PMID", "G2P_IDs"]
    for row in rows:
        assert row["PMID"] and row["G2P_IDs"]


def test_score_cutoff_monotone(tmp_path, fixtures_present):
    """Lower score_cutoff yields >= the rows of a higher cutoff."""
    out_high = tmp_path / "out_high.csv"
    out_low = tmp_path / "out_low.csv"
    run_cleaner(out_high, score_cutoff=0.9)
    run_cleaner(out_low, score_cutoff=0.0)
    n_high = len(read_output(out_high))
    n_low = len(read_output(out_low))
    assert n_low >= n_high


def test_gene_match_applies_with_zero_cutoff(tmp_path, fixtures_present):
    """The bug Diana introduced: with --score_cutoff 0 the gene-match check
    must STILL be applied. We verify it by confirming every output row's G2P
    gene appears in the PMID's gene2pubtator mentions."""
    out = tmp_path / "out_zero_cutoff.csv"
    run_cleaner(out, score_cutoff=0.0)

    # Build PMID -> set of canonical symbols (mirroring the cleaner).
    geneid_to_sym: dict[str, str] = {}
    with gzip.open(GENE_INFO, "rt") as f:
        header = f.readline().rstrip("\n").lstrip("#").split("\t")
        i_tax, i_gid, i_sym = header.index("tax_id"), header.index("GeneID"), header.index("Symbol")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(i_tax, i_gid, i_sym):
                continue
            if parts[i_tax] == "9606":
                geneid_to_sym[parts[i_gid]] = parts[i_sym]

    pmid_to_syms: dict[str, set[str]] = {}
    with gzip.open(GENE2PUBTATOR, "rt") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            eid = parts[2].split(";")[0].strip() if parts[2] else ""
            sym = geneid_to_sym.get(eid)
            if sym:
                pmid_to_syms.setdefault(parts[0], set()).add(sym)

    g2p_syms: dict[str, list[str]] = {}
    with open(G2P, newline="") as f:
        for row in csv.DictReader(f):
            gid = (row.get("g2p id") or "").strip()
            if not gid:
                continue
            gene = (row.get("gene symbol") or "").strip()
            prev = (row.get("previous gene symbols") or "").strip()
            syms = [gene] if gene else []
            syms += [p.strip() for p in prev.split(";") if p.strip()]
            if syms:
                g2p_syms[gid] = syms

    rows = read_output(out)
    assert len(rows) > 0, "expected at least one surviving row at score_cutoff=0"
    for row in rows:
        pmid, gid = row["PMID"], row["G2P_IDs"]
        mentioned = pmid_to_syms.get(pmid, set())
        candidate_syms = g2p_syms.get(gid, [])
        assert any(s in mentioned for s in candidate_syms), (
            f"row PMID={pmid} G2P={gid} survived but no G2P gene "
            f"({candidate_syms}) in pubtator mentions {mentioned}"
        )


def test_no_hallucinated_g2p_ids(tmp_path, fixtures_present):
    """Every output G2P_ID must exist in the G2P CSV."""
    out = tmp_path / "out.csv"
    run_cleaner(out, score_cutoff=0.0)
    valid = set()
    with open(G2P, newline="") as f:
        for row in csv.DictReader(f):
            gid = (row.get("g2p id") or "").strip()
            if gid:
                valid.add(gid)
    rows = read_output(out)
    for row in rows:
        assert row["G2P_IDs"] in valid, f"hallucinated G2P_ID survived: {row['G2P_IDs']}"


def test_pmid_typing(tmp_path, fixtures_present):
    """Every output PMID parses as int and is a real PMID from the input parquet."""
    out = tmp_path / "out.csv"
    run_cleaner(out, score_cutoff=0.0)
    rows = read_output(out)
    parquet_pmids = {str(p) for p in pq.read_table(LLM, columns=["pmid"]).column(0).to_pylist()}
    for row in rows:
        int(row["PMID"])  # must parse
        assert row["PMID"] in parquet_pmids


def test_streaming_memory(tmp_path, fixtures_present):
    """The merged cleaner uses streaming parquet — peak RSS stays small.
    Generous 1 GB ceiling to cover Python/pyarrow startup overhead."""
    out = tmp_path / "out.csv"
    cmd = [
        "/usr/bin/env", "bash", "-c",
        "exec " + " ".join([
            sys.executable, str(SCRIPT),
            "--llm_file", str(LLM),
            "--g2p_file", str(G2P),
            "--gene2pubtator", str(GENE2PUBTATOR),
            "--gene_info", str(GENE_INFO),
            "--output_csv", str(out),
            "--score_cutoff", "0.5",
        ]),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    # Use getrusage of *this* process as a baseline; the subprocess cost is
    # dominated by Python+pyarrow startup, ~150-300 MB. Cap at 1 GB.
    peak_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    # On Linux ru_maxrss is in KB
    assert peak_kb < 1_000_000, f"peak child RSS {peak_kb} KB > 1 GB"


def test_no_gene_check_retains_more(tmp_path, fixtures_present):
    """--no_gene_check (R2-C1/R3.4) relaxes the gene-mention filter, so it can only
    RETAIN more mappings than the default, and reports the attrition it would impose."""
    on = tmp_path / "gene_on.csv"
    off = tmp_path / "gene_off.csv"
    run_cleaner(on, score_cutoff=0.9)
    res = run_cleaner(off, score_cutoff=0.9, no_gene_check=True)
    n_on, n_off = len(read_output(on)), len(read_output(off))
    assert n_off >= n_on, "relaxing the gene filter must not drop mappings"
    assert n_off > n_on, "fixtures should include gene-filtered mappings to retain"
    assert "Gene-filter" in res.stdout  # attrition is reported for R2-C1/R3.4
