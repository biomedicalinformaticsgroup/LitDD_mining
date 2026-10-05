"""Unit tests for the deterministic parts of the adjudication stage: prompt construction
(``litdd.pipeline.llm_prompt``), answer parsing (``litdd.pipeline.llm_answer``) and row
striping (``litdd.pipeline.llm_map``). No vLLM or GPU is needed.
"""
from __future__ import annotations

import io
import json

import numpy as np
import pytest

from litdd.pipeline import llm_answer, llm_map, llm_prompt


# --------------------------------------------------------------------------------------
# Answer extraction
# --------------------------------------------------------------------------------------
def test_extract_last_answer_single():
    assert llm_answer.extract_last_answer("reasoning...\nANSWER: G2P123") == "G2P123"


def test_extract_last_answer_takes_last_when_multiple():
    txt = "ANSWER: G2P1\n...more reasoning...\nANSWER: G2P9;G2P8"
    assert llm_answer.extract_last_answer(txt) == "G2P9;G2P8"


def test_extract_last_answer_no_match_string():
    assert llm_answer.extract_last_answer("ANSWER: NO MATCH") == "NO MATCH"


def test_extract_last_answer_none_when_absent_or_empty():
    assert llm_answer.extract_last_answer("no answer line here") is None
    assert llm_answer.extract_last_answer("") is None
    assert llm_answer.extract_last_answer(None) is None


def test_extract_last_answer_handles_harmony_glue_and_case():
    """GPT-OSS harmony output concatenates channels: '...assistantfinalANSWER: NO MATCH'."""
    txt = "analysisThe gene is EFTUD2 ... ANSWER: G2P01236 would fit.assistantfinalANSWER: NO MATCH"
    assert llm_answer.extract_last_answer(txt) == "NO MATCH"
    assert llm_answer.extract_last_answer("answer: G2P00001") == "G2P00001"


# --------------------------------------------------------------------------------------
# Answer parsing
# --------------------------------------------------------------------------------------
def test_parse_answer_schema_cases():
    ok = llm_answer.parse_answer("G2P01236", ["G2P01236", "G2P01399"])
    assert ok == {"llm_dis_map": "G2P01236", "answer_format_valid": True,
                  "answer_uncertain": False, "answer_ids_in_candidates": True}
    multi = llm_answer.parse_answer("G2P01236;G2P01399", ["G2P01236", "G2P01399"])
    assert multi["llm_dis_map"] == "G2P01236;G2P01399" and multi["answer_format_valid"]
    nm = llm_answer.parse_answer("NO MATCH", ["G2P01236"])
    assert nm["llm_dis_map"] == "NO MATCH" and nm["answer_format_valid"]
    assert nm["answer_ids_in_candidates"] is None


def test_parse_answer_recovers_decorated_ids_but_flags_format():
    p = llm_answer.parse_answer("**G2P01236** (EFTUD2).", ["G2P01236"])
    assert p["llm_dis_map"] == "G2P01236"
    assert p["answer_format_valid"] is False
    assert p["answer_ids_in_candidates"] is True
    # duplicates collapse, order kept
    assert llm_answer.parse_answer("G2P2; G2P1; G2P2", None)["llm_dis_map"] == "G2P2;G2P1"


def test_parse_answer_flags_hallucination_and_uncertain():
    h = llm_answer.parse_answer("G2P99999", ["G2P00001"])
    assert h["llm_dis_map"] == "G2P99999" and h["answer_ids_in_candidates"] is False
    u = llm_answer.parse_answer("UNCERTAIN", ["G2P00001"])
    assert u["llm_dis_map"] == "NO MATCH" and u["answer_uncertain"] and u["answer_format_valid"]
    none = llm_answer.parse_answer(None, ["G2P00001"])
    assert none["llm_dis_map"] is None and none["answer_format_valid"] is False
    assert llm_answer.parse_answer("I cannot decide", ["G2P00001"])["llm_dis_map"] is None


def test_restrict_to_panel_keeps_released_ids_only():
    final = {"G2P1", "G2P3"}
    assert llm_answer.restrict_to_panel("G2P1;G2P2", final) == "G2P1"
    assert llm_answer.restrict_to_panel("G2P2", final) == "NO MATCH"
    assert llm_answer.restrict_to_panel("G2P1;G2P3", final) == "G2P1;G2P3"
    # NO MATCH, None and an absent panel pass through unchanged
    assert llm_answer.restrict_to_panel("NO MATCH", final) == "NO MATCH"
    assert llm_answer.restrict_to_panel(None, final) is None
    assert llm_answer.restrict_to_panel("G2P2", None) == "G2P2"


# --------------------------------------------------------------------------------------
# Prompt rendering
# --------------------------------------------------------------------------------------
def test_build_llm_prompt_includes_tiab_and_numbered_candidates():
    prompt = llm_prompt.build_llm_prompt(
        "My TIAB text", ["G2P1 - GENEA - foo", "G2P2 - GENEB - bar"]
    )
    assert "My TIAB text" in prompt
    assert "1) G2P1 - GENEA - foo" in prompt
    assert "2) G2P2 - GENEB - bar" in prompt
    assert "ANSWER:" in prompt  # output schema present


def test_build_llm_prompt_rejects_empty_candidates():
    """An empty candidate list raises instead of rendering a candidate-free prompt, whose
    only possible answer (NO MATCH) would be indistinguishable from a negative."""
    with pytest.raises(ValueError, match="no candidate threads"):
        llm_prompt.build_llm_prompt("tiab", [])


def test_build_llm_prompt_states_actual_candidate_count():
    """The prompt describes however many candidates it was given."""
    for n in (1, 3, 5, 8):
        cands = [f"G2P{i:05d} - GENE{i} - disease {i}" for i in range(n)]
        prompt = llm_prompt.build_llm_prompt("tiab", cands)
        assert f"{n} candidate LGMDE" in prompt
        assert f"numbered 1-{n}" in prompt
        assert f"Only choose from the {n} candidate(s)" in prompt
        # every candidate is numbered, so multi-line threads have clear boundaries
        for i in range(n):
            assert f"{i + 1}) {cands[i]}" in prompt
        assert "the 5 candidates" not in prompt


def test_build_llm_prompt_singular_for_one_candidate():
    prompt = llm_prompt.build_llm_prompt("tiab", ["G2P00001 - GENE - disease"])
    assert "1 candidate LGMDE thread," in prompt


def test_prompt_file_is_the_rendered_prompt():
    """The prompt file is rendered as is: template head, no unfilled placeholders, schema tail."""
    template = llm_prompt.load_prompt_template()
    prompt = llm_prompt.build_llm_prompt("TIAB X", ["G2P00001 - A - x", "G2P00002 - B - y"])
    head = template.split("{n}")[0]
    assert prompt.startswith(head)
    assert "{tiab}" not in prompt and "{candidate_lines}" not in prompt
    assert prompt.rstrip().endswith("Return exactly one line in the schema above.")
    assert "\n        You are an expert" not in prompt
    assert "\nYou are an expert" in prompt


def test_default_prompt_is_the_decision_rubric_and_renders_the_contract():
    """The default prompt is prompts/decision_rubric.txt: an explicit disorder test, former
    gene symbols, entries from other G2P panels, and the one-line ANSWER schema."""
    assert llm_prompt.DEFAULT_PROMPT_FILE.endswith("decision_rubric.txt")
    text = llm_prompt.load_prompt_template()
    assert "Prefer selecting at least one candidate over NO MATCH" not in text
    assert "consider returning the matching candidate anyway" not in text
    assert "It is a CLINICALLY DISTINCT disorder only when one of the following holds" in text
    assert "previous gene symbols" in text
    assert "including one from a different G2P panel" in text
    prompt = llm_prompt.build_llm_prompt("TIAB", ["G2P1 - A - x", "G2P2 - B - y"])
    assert "Only choose from the 2 candidate(s)" in prompt and "ANSWER: NO MATCH" in prompt
    assert "{" not in prompt.replace("{}", "")


def test_load_prompt_template_requires_both_placeholders(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text("{tiab} only")
    with pytest.raises(ValueError, match="candidate_lines"):
        llm_prompt.load_prompt_template(str(bad))
    bad.write_text("{candidate_lines} only")
    with pytest.raises(ValueError, match="tiab"):
        llm_prompt.load_prompt_template(str(bad))


def test_fill_placeholders_leaves_other_braces():
    out = llm_prompt.fill_placeholders("{n} {x} {\"json\": 1}", {"n": 2, "unused": 0})
    assert out == "2 {x} {\"json\": 1}"


# --------------------------------------------------------------------------------------
# Candidate cells
# --------------------------------------------------------------------------------------
def test_candidate_list_normalises_cells():
    assert llm_prompt.candidate_list(["G2P1", " G2P2 ", "", None]) == ["G2P1", "G2P2"]
    assert llm_prompt.candidate_list(np.array(["G2P1", "G2P2"], dtype=object)) == ["G2P1", "G2P2"]
    assert llm_prompt.candidate_list(("G2P3",)) == ["G2P3"]
    assert llm_prompt.candidate_list(None) == []
    assert llm_prompt.candidate_list(float("nan")) == []
    assert llm_prompt.candidate_list("G2P1") == []   # a bare string is not a list


def test_candidate_list_accepts_pyarrow_scalars():
    import pyarrow as pa
    arr = pa.array([["G2P1", "G2P2"], []])
    assert llm_prompt.candidate_list(arr[0]) == ["G2P1", "G2P2"]
    assert llm_prompt.candidate_list(arr[1]) == []


def test_candidate_ids_from_flat_and_context_threads():
    flat = "G2P01236 - EFTUD2 - 603892.0 - ..."
    ctx = "G2P ID: G2P01399\nGene Symbol: CHD7\nDisease Name: CHARGE"
    assert llm_prompt.candidate_ids([flat, ctx, "no id here"]) == ["G2P01236", "G2P01399", None]


# --------------------------------------------------------------------------------------
# Contextualised threads
# --------------------------------------------------------------------------------------
def test_contextualise_swaps_by_id_and_counts_misses():
    ctx = {"G2P01236": "G2P ID: G2P01236\nGene Symbol: EFTUD2"}
    counter = {}
    out = llm_prompt.contextualise(["G2P01236", "G2P09999"], ctx, counter)
    assert out == ["G2P ID: G2P01236\nGene Symbol: EFTUD2", "G2P09999"]
    assert counter == {"missing": 1}


def test_load_context_threads_drops_literal_none_lines(tmp_path):
    p = tmp_path / "ctx.json"
    p.write_text(json.dumps({
        "__provenance__": {"panel": "x"},
        "G2P00001": "G2P ID: G2P00001\nGene Symbol: A\nDisease Synonyms: None\nDisease Definition: None\nPhenotypes: a; b\n",
    }))
    ctx = llm_prompt.load_context_threads(str(p))
    assert list(ctx) == ["G2P00001"]
    assert ctx["G2P00001"] == "G2P ID: G2P00001\nGene Symbol: A\nPhenotypes: a; b"


def test_drop_context_fields_removes_named_lines_only():
    ctx = {"G2P1": "G2P ID: G2P1\nPrevious Gene Symbols: CADASIL\nDisease Definition: x\nPhenotypes: a; b",
           "G2P2": "G2P ID: G2P2\nPhenotypes: c"}
    out = llm_prompt.drop_context_fields(ctx, "Disease Definition, Phenotypes")
    assert out["G2P1"] == "G2P ID: G2P1\nPrevious Gene Symbols: CADASIL"
    assert out["G2P2"] == "G2P ID: G2P2"
    # a field name that is a prefix of another label does not match it
    assert llm_prompt.drop_context_fields(ctx, "Disease")["G2P1"] == ctx["G2P1"]


# --------------------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------------------
ALL_PANEL_CSV = ("g2p id,gene symbol,panel\n"
                 "G2P1,NOTCH3,DD\nG2P2,NOTCH3,Cardiac\nG2P3,CHD7,DD\nG2P4,CHD7,Ear\nG2P5,MECP2,DD\n")


def _siblings():
    return llm_prompt.load_panel_siblings(io.StringIO(ALL_PANEL_CSV))


def test_load_panel_siblings_indexes_the_export():
    sib = _siblings()
    assert sib["gene"] == {"G2P1": "NOTCH3", "G2P2": "NOTCH3", "G2P3": "CHD7", "G2P4": "CHD7", "G2P5": "MECP2"}
    assert sib["ids"]["NOTCH3"] == ["G2P1", "G2P2"] and sib["ids"]["CHD7"] == ["G2P3", "G2P4"]
    assert sib["panel"]["G2P2"] == "Cardiac"


def test_add_panel_siblings_appends_other_entries_of_the_candidate_genes_in_order():
    sib = _siblings()
    assert llm_prompt.add_panel_siblings(["G2P3", "G2P1"], sib) == ["G2P3", "G2P1", "G2P4", "G2P2"]
    # already-present entries are not duplicated; unknown ids add nothing
    assert llm_prompt.add_panel_siblings(["G2P1", "G2P2"], sib) == ["G2P1", "G2P2"]
    assert llm_prompt.add_panel_siblings(["G2P999"], sib) == ["G2P999"]
    assert llm_prompt.add_panel_siblings([], sib) == []


def test_label_panel_appends_the_panel_line():
    sib = _siblings()
    assert llm_prompt.label_panel("G2P ID: G2P2\nGene Symbol: NOTCH3", sib) == \
        "G2P ID: G2P2\nGene Symbol: NOTCH3\nG2P Panel: Cardiac"
    assert llm_prompt.label_panel("G2P999", sib) == "G2P999"
    assert llm_prompt.label_panel("no id", sib) == "no id"


def test_prompt_path_from_ids_to_prompt():
    """The driver's prompt path: ids -> siblings -> context blocks -> panel label -> prompt."""
    sib = _siblings()
    ctx = llm_prompt.drop_context_fields(
        {"G2P1": "G2P ID: G2P1\nGene Symbol: NOTCH3\nDisease Definition: d\nPhenotypes: p",
         "G2P2": "G2P ID: G2P2\nGene Symbol: NOTCH3\nDisease Name: CADASIL"},
        "Disease Definition,Phenotypes")
    labs = llm_prompt.add_panel_siblings(llm_prompt.candidate_list(["G2P1"]), sib)
    blocks = [llm_prompt.label_panel(b, sib) for b in llm_prompt.contextualise(labs, ctx)]
    assert blocks == ["G2P ID: G2P1\nGene Symbol: NOTCH3\nG2P Panel: DD",
                      "G2P ID: G2P2\nGene Symbol: NOTCH3\nDisease Name: CADASIL\nG2P Panel: Cardiac"]
    prompt = llm_prompt.build_llm_prompt("A CADASIL family.", blocks)
    assert "2 candidate LGMDE threads, numbered 1-2" in prompt
    assert "1) G2P ID: G2P1\nGene Symbol: NOTCH3\nG2P Panel: DD\n2) G2P ID: G2P2" in prompt
    assert llm_prompt.candidate_ids(labs) == ["G2P1", "G2P2"]


# --------------------------------------------------------------------------------------
# Driver configuration and row striping
# --------------------------------------------------------------------------------------
def test_parse_args_builds_config_with_flag_names():
    cfg = llm_map.parse_args(["--shards_dir", "s", "--llm_model", "m", "--context_json", "c.json",
                              "--reasoning_effort", "none", "--num_shards", "4", "--shard_index", "1"])
    assert isinstance(cfg, llm_map.LlmMapConfig)
    assert cfg.shards_dir == "s" and cfg.llm_model == "m" and cfg.context_json == "c.json"
    assert cfg.reasoning_effort is None and cfg.num_shards == 4 and cfg.shard_index == 1
    assert cfg.prompt_file == llm_prompt.DEFAULT_PROMPT_FILE
    with pytest.raises(SystemExit):
        llm_map.parse_args(["--shards_dir", "s", "--llm_model", "m"])   # --context_json required


def test_row_slice_partitions_every_row_exactly_once():
    """Every row is claimed by exactly one worker, for any row:worker ratio."""
    for n_rows in (0, 1, 7, 100, 195558):
        for num_shards in (1, 2, 3, 8, 16):
            claimed = [i for s in range(num_shards)
                       for i in llm_map.row_slice_for_worker(n_rows, s, num_shards)]
            assert sorted(claimed) == list(range(n_rows))


def test_row_slice_gives_every_worker_work_when_workers_exceed_files():
    """More workers than shard files still gives every worker rows."""
    n_rows, num_shards = 1000, 8
    for s in range(num_shards):
        assert llm_map.row_slice_for_worker(n_rows, s, num_shards), f"worker {s} got no rows"


def test_row_slice_is_balanced():
    counts = [len(llm_map.row_slice_for_worker(1000, s, 8)) for s in range(8)]
    assert max(counts) - min(counts) <= 1


def test_row_slice_no_sharding_returns_everything():
    assert llm_map.row_slice_for_worker(5, None, None) == [0, 1, 2, 3, 4]
    assert llm_map.row_slice_for_worker(5, 0, 1) == [0, 1, 2, 3, 4]
