"""Unit tests for the deterministic logic in `litdd/pipeline/llm_map.py`:
LLM-output parsing, prompt construction, batching and shard selection.

These exercise pure functions only (no vLLM/GPU). Run with
`pytest tests/test_llm_map.py -v`.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "litdd" / "pipeline"))

import llm_map  # noqa: E402
import pytest  # noqa: E402


def test_extract_last_answer_single():
    assert llm_map.extract_last_answer("reasoning...\nANSWER: G2P123") == "G2P123"


def test_extract_last_answer_takes_last_when_multiple():
    txt = "ANSWER: G2P1\n...more reasoning...\nANSWER: G2P9;G2P8"
    assert llm_map.extract_last_answer(txt) == "G2P9;G2P8"


def test_extract_last_answer_no_match_string():
    assert llm_map.extract_last_answer("ANSWER: NO MATCH") == "NO MATCH"


def test_extract_last_answer_none_when_absent_or_empty():
    assert llm_map.extract_last_answer("no answer line here") is None
    assert llm_map.extract_last_answer("") is None
    assert llm_map.extract_last_answer(None) is None


def test_build_llm_prompt_includes_tiab_and_numbered_candidates():
    prompt = llm_map.build_llm_prompt(
        "My TIAB text", ["G2P1 - GENEA - foo", "G2P2 - GENEB - bar"]
    )
    assert "My TIAB text" in prompt
    assert "1) G2P1 - GENEA - foo" in prompt
    assert "2) G2P2 - GENEB - bar" in prompt
    assert "ANSWER:" in prompt  # output schema present


def test_build_llm_prompt_rejects_empty_candidates():
    """An empty candidate list must fail loudly, not render a candidate-free prompt.

    Previously this returned a prompt listing no candidates at all; the model would answer
    NO MATCH every time, which is indistinguishable from a genuine negative. At corpus
    scale that turns a broken candidate join into a plausible-looking result set.
    """
    with pytest.raises(ValueError, match="no candidate threads"):
        llm_map.build_llm_prompt("tiab", [])


def test_build_llm_prompt_states_actual_candidate_count():
    """The prompt must describe however many candidates it was given, not a fixed 5."""
    for n in (1, 3, 5, 8):
        cands = [f"G2P{i:05d} - GENE{i} - disease {i}" for i in range(n)]
        prompt = llm_map.build_llm_prompt("tiab", cands)
        assert f"{n} candidate LGMDE" in prompt
        assert f"numbered 1-{n}" in prompt
        assert f"Only choose from the {n} candidate(s)" in prompt
        # every candidate is numbered, so multi-line threads have clear boundaries
        for i in range(n):
            assert f"{i + 1}) {cands[i]}" in prompt
        assert "the 5 candidates" not in prompt


def test_build_llm_prompt_singular_for_one_candidate():
    prompt = llm_map.build_llm_prompt("tiab", ["G2P00001 - GENE - disease"])
    assert "1 candidate LGMDE thread," in prompt


def test_prompt_file_is_the_rendered_prompt():
    """The vendored prompt file (the R3.8 appendix) is what the pipeline actually sends."""
    template = llm_map.load_prompt_template()
    prompt = llm_map.build_llm_prompt("TIAB X", ["G2P00001 - A - x", "G2P00002 - B - y"])
    head = template.split("{n}")[0]
    assert prompt.startswith(head)
    assert "{tiab}" not in prompt and "{candidate_lines}" not in prompt
    assert prompt.rstrip().endswith("Return exactly one line in the schema above.")
    # no stray f-string indentation survives in the rendered prompt
    assert "\n        You are an expert" not in prompt
    assert "\nYou are an expert" in prompt


def test_prompt_matches_upstream_rubric_when_available():
    """Byte-compare the rubric against Fabian's original_paper.txt if the checkout exists.

    The only intended differences: the candidate-count sentences ({n}, numbered) and the
    candidate insertion. The decision rubric itself must be identical, otherwise benchmark
    numbers from the harness do not transfer to the pipeline.
    """
    upstream = ROOT / "upgraded-octo-happiness" / "prompt" / "baseline" / "original_paper.txt"
    if not upstream.exists():
        pytest.skip("upstream harness not checked out")
    ours = llm_map.load_prompt_template()
    theirs = upstream.read_text(encoding="utf-8")
    def rubric(text):
        start = text.index("Decision rubric")
        end = text.index("Output schema")
        return "\n".join(ln.rstrip() for ln in text[start:end].splitlines())
    assert rubric(ours) == rubric(theirs)


def test_extract_last_answer_handles_harmony_glue_and_case():
    """GPT-OSS harmony output concatenates channels: '...assistantfinalANSWER: NO MATCH'."""
    txt = "analysisThe gene is EFTUD2 ... ANSWER: G2P01236 would fit.assistantfinalANSWER: NO MATCH"
    assert llm_map.extract_last_answer(txt) == "NO MATCH"
    assert llm_map.extract_last_answer("answer: G2P00001") == "G2P00001"


def test_parse_answer_schema_cases():
    ok = llm_map.parse_answer("G2P01236", ["G2P01236", "G2P01399"])
    assert ok == {"llm_dis_map": "G2P01236", "answer_format_valid": True,
                  "answer_uncertain": False, "answer_ids_in_candidates": True}
    multi = llm_map.parse_answer("G2P01236;G2P01399", ["G2P01236", "G2P01399"])
    assert multi["llm_dis_map"] == "G2P01236;G2P01399" and multi["answer_format_valid"]
    nm = llm_map.parse_answer("NO MATCH", ["G2P01236"])
    assert nm["llm_dis_map"] == "NO MATCH" and nm["answer_format_valid"]
    assert nm["answer_ids_in_candidates"] is None


def test_parse_answer_recovers_decorated_ids_but_flags_format():
    p = llm_map.parse_answer("**G2P01236** (EFTUD2).", ["G2P01236"])
    assert p["llm_dis_map"] == "G2P01236"
    assert p["answer_format_valid"] is False
    assert p["answer_ids_in_candidates"] is True
    # duplicates collapse, order kept
    assert llm_map.parse_answer("G2P2; G2P1; G2P2", None)["llm_dis_map"] == "G2P2;G2P1"


def test_parse_answer_flags_hallucination_and_uncertain():
    h = llm_map.parse_answer("G2P99999", ["G2P00001"])
    assert h["llm_dis_map"] == "G2P99999" and h["answer_ids_in_candidates"] is False
    u = llm_map.parse_answer("UNCERTAIN", ["G2P00001"])
    assert u["llm_dis_map"] == "NO MATCH" and u["answer_uncertain"] and u["answer_format_valid"]
    none = llm_map.parse_answer(None, ["G2P00001"])
    assert none["llm_dis_map"] is None and none["answer_format_valid"] is False
    assert llm_map.parse_answer("I cannot decide", ["G2P00001"])["llm_dis_map"] is None


def test_candidate_ids_from_flat_and_context_threads():
    flat = "G2P01236 - EFTUD2 - 603892.0 - ..."
    ctx = "G2P ID: G2P01399\nGene Symbol: CHD7\nDisease Name: CHARGE"
    assert llm_map.candidate_ids([flat, ctx, "no id here"]) == ["G2P01236", "G2P01399", None]


def test_contextualise_swaps_by_id_and_counts_misses():
    ctx = {"G2P01236": "G2P ID: G2P01236\nGene Symbol: EFTUD2"}
    counter = {}
    out = llm_map.contextualise(["G2P01236 - EFTUD2 - x", "G2P09999 - ZZZ - y"], ctx, counter)
    assert out == ["G2P ID: G2P01236\nGene Symbol: EFTUD2", "G2P09999 - ZZZ - y"]
    assert counter == {"missing": 1}


def test_load_context_threads_drops_literal_none_lines(tmp_path):
    import json
    p = tmp_path / "ctx.json"
    p.write_text(json.dumps({
        "__provenance__": {"panel": "x"},
        "G2P00001": "G2P ID: G2P00001\nGene Symbol: A\nDisease Synonyms: None\nDisease Definition: None\nPhenotypes: a; b\n",
    }))
    ctx = llm_map.load_context_threads(str(p))
    assert list(ctx) == ["G2P00001"]
    assert ctx["G2P00001"] == "G2P ID: G2P00001\nGene Symbol: A\nPhenotypes: a; b"


def test_to_labels_min_score_gates_before_the_llm():
    cell = [{"label": "G2P1 - a", "score": 0.97}, {"label": "G2P2 - a", "score": 0.42},
            {"label": "G2P3 - b", "score": 0.91}, ("G2P4 - c", 0.05), "G2P5 - unscored"]
    assert llm_map.to_labels(cell, min_score=0.9) == ["G2P1 - a", "G2P3 - b", "G2P5 - unscored"]
    assert llm_map.to_labels(cell, max_candidates=1, min_score=0.9) == ["G2P1 - a"]
    assert llm_map.to_labels([{"label": "G2P9 - z", "score": 0.1}], min_score=0.9) == []
    # no gate -> everything kept, order preserved
    assert len(llm_map.to_labels(cell)) == 5


def test_to_labels_caps_only_when_asked():
    cell = [{"label": "G2P1 - a", "score": 0.9}, ("G2P2 - b", 0.5), "G2P3 - c"]
    assert llm_map.to_labels(cell) == ["G2P1 - a", "G2P2 - b", "G2P3 - c"]
    assert llm_map.to_labels(cell, max_candidates=2) == ["G2P1 - a", "G2P2 - b"]
    assert llm_map.to_labels(None) == []
    assert llm_map.to_labels('[["G2P1 - a", 0.9]]') == ["G2P1 - a"]


def test_batched_indices():
    assert list(llm_map.batched_indices(0, 5, 2)) == [(0, 2), (2, 4), (4, 5)]
    assert list(llm_map.batched_indices(0, 0, 2)) == []
    assert list(llm_map.batched_indices(3, 7, 10)) == [(3, 7)]


def test_select_shards_for_worker():
    paths = [f"s{i}.parquet" for i in range(6)]
    assert llm_map.select_shards_for_worker(paths, 0, 3) == ["s0.parquet", "s3.parquet"]
    assert llm_map.select_shards_for_worker(paths, 2, 3) == ["s2.parquet", "s5.parquet"]
    # no sharding -> all paths
    assert llm_map.select_shards_for_worker(paths, None, None) == paths


def test_row_slice_partitions_every_row_exactly_once():
    """Every row is claimed by exactly one worker, for any file:worker ratio.

    The deployed run sharded by *file index* with 4 files and 8 workers, so workers 4-7 got
    nothing and half an 8x A100 allocation idled for six days. Row striping cannot do that.
    """
    for n_rows in (0, 1, 7, 100, 195558):
        for num_shards in (1, 2, 3, 8, 16):
            claimed = [i for s in range(num_shards)
                       for i in llm_map.row_slice_for_worker(n_rows, s, num_shards)]
            assert sorted(claimed) == list(range(n_rows))


def test_row_slice_gives_every_worker_work_when_workers_exceed_files():
    """The exact condition that idled four GPUs: more workers than shard files."""
    n_rows, num_shards = 1000, 8
    for s in range(num_shards):
        assert llm_map.row_slice_for_worker(n_rows, s, num_shards), f"worker {s} got no rows"


def test_row_slice_is_balanced():
    counts = [len(llm_map.row_slice_for_worker(1000, s, 8)) for s in range(8)]
    assert max(counts) - min(counts) <= 1


def test_row_slice_no_sharding_returns_everything():
    assert llm_map.row_slice_for_worker(5, None, None) == [0, 1, 2, 3, 4]
    assert llm_map.row_slice_for_worker(5, 0, 1) == [0, 1, 2, 3, 4]
