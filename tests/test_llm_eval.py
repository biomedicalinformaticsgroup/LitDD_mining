"""Tests for litdd.evaluation.llm_adjudication_eval and litdd.evaluation.common on a
six-abstract synthetic frame whose counts can be checked by hand, including abstracts with
several gold ids and answers with several ids."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from litdd.evaluation import common
from litdd.evaluation import llm_adjudication_eval as ev

GENES = {"G2P1": "A", "G2P2": "A", "G2P3": "D", "G2P4": "E", "G2P5": "F", "G2P7": "C", "G2P9": "B"}


def row(row_id, answer, candidates, **flags):
    base = {"row_id": row_id, "llm_dis_map": answer, "candidates": list(candidates),
            "generated_text": "x", "answer_format_valid": True, "answer_uncertain": False,
            "answer_ids_in_candidates": True, "finish_reason": "stop", "gen_tokens": 100,
            "prompt_tokens": 1000}
    base.update(flags)
    return base


@pytest.fixture()
def toy():
    gold = pd.DataFrame([
        {"row_id": "a", "pmid": "1", "true_g2p_ids": "G2P1", "n_gold": 1, "genereviews": False, "bert_predict": 1},
        {"row_id": "b", "pmid": "2", "true_g2p_ids": "G2P1;G2P2", "n_gold": 2, "genereviews": False, "bert_predict": 1},
        {"row_id": "c", "pmid": "3", "true_g2p_ids": "", "n_gold": 0, "genereviews": False, "bert_predict": 1},
        {"row_id": "d", "pmid": "4", "true_g2p_ids": "G2P3", "n_gold": 1, "genereviews": False, "bert_predict": 0},
        {"row_id": "e", "pmid": "5", "true_g2p_ids": "", "n_gold": 0, "genereviews": False, "bert_predict": 0},
        {"row_id": "f", "pmid": "6", "true_g2p_ids": "G2P5", "n_gold": 1, "genereviews": False, "bert_predict": 1},
    ])
    llm = pd.DataFrame([
        row("a", "G2P1", ["G2P1", "G2P9"]),
        row("b", "G2P1;G2P9", ["G2P1", "G2P2", "G2P9"], gen_tokens=300),
        row("c", "NO MATCH", ["G2P7"], answer_ids_in_candidates=None, gen_tokens=50),
        row("d", "G2P3", ["G2P3"], gen_tokens=120),
        row("e", "G2P8", ["G2P4"], answer_ids_in_candidates=False, finish_reason="length", gen_tokens=8192),
        row("f", None, ["G2P5"], answer_format_valid=False, answer_ids_in_candidates=None, gen_tokens=10),
    ])
    pairs = pd.DataFrame([
        {"row_id": "a", "g2p_id": "G2P1", "label": 1, "in_candidates": True},
        {"row_id": "a", "g2p_id": "G2P9", "label": 0, "in_candidates": True},
        {"row_id": "b", "g2p_id": "G2P1", "label": 1, "in_candidates": True},
        {"row_id": "b", "g2p_id": "G2P2", "label": 1, "in_candidates": True},
        {"row_id": "c", "g2p_id": "G2P7", "label": 0, "in_candidates": True},
        {"row_id": "d", "g2p_id": "G2P3", "label": 1, "in_candidates": True},
        {"row_id": "e", "g2p_id": "G2P4", "label": 0, "in_candidates": True},
        {"row_id": "f", "g2p_id": "G2P5", "label": 1, "in_candidates": True},
        {"row_id": "f", "g2p_id": "G2P6", "label": 1, "in_candidates": False},
    ])
    return llm, gold, pairs


def test_per_tiab_table_sets_and_flags(toy):
    llm, gold, _ = toy
    t = ev.per_tiab_table(llm, gold, genes=GENES).set_index("row_id")
    assert bool(t.loc["a", "exact_correct"]) and not bool(t.loc["b", "exact_correct"])
    assert (t.loc["b", "id_tp"], t.loc["b", "id_fp"], t.loc["b", "id_fn"]) == (1, 1, 1)
    assert t.loc["b", "multi_gold"] and t.loc["b", "cands_share_gene"]
    assert not t.loc["a", "cands_share_gene"]
    assert t.loc["c", "exact_correct"] and t.loc["c", "no_match"]
    assert t.loc["d", "id_tp_screen"] == 0 and t.loc["a", "id_tp_screen"] == 1
    assert t.loc["e", "hallucinated"] and t.loc["e", "truncated"]
    assert t.loc["f", "unparsed"] and t.loc["f", "format_invalid"]
    assert t.loc["b", "n_candidates"] == 3


def test_id_micro_counts_multi_disease(toy):
    llm, gold, _ = toy
    t = ev.per_tiab_table(llm, gold, genes=GENES)
    v = ev.view_id_micro(t)
    assert (v["tp"], v["fp"], v["fn"]) == (3, 2, 2)
    assert v["precision"] == 0.6 and v["recall"] == 0.6
    lo, hi = v["precision_ci95"]
    assert lo < 0.6 < hi
    s = ev.view_id_micro(t, "_screen")
    assert (s["tp"], s["fp"], s["fn"]) == (2, 1, 3)


def test_tiab_exact_view(toy):
    llm, gold, _ = toy
    t = ev.per_tiab_table(llm, gold)
    v = ev.view_tiab_exact(t)
    assert (v["tp"], v["fp"], v["fn"], v["tn"]) == (2, 2, 2, 1)
    assert v["accuracy"] == 0.5


def test_end_to_end_view_applies_the_screen(toy):
    llm, gold, _ = toy
    t = ev.per_tiab_table(llm, gold)
    v = ev.view_end_to_end_exact(t)
    assert (v["tp"], v["fp"], v["fn"], v["tn"]) == (1, 1, 3, 2)
    assert v["n_abstracts"] == 6


def test_pair_level_view_counts_gate_misses(toy):
    llm, gold, pairs = toy
    t = ev.per_tiab_table(llm, gold)
    v = ev.view_pair_level(t, pairs)
    assert (v["tp"], v["fp"], v["fn"], v["tn"]) == (3, 0, 2, 3)
    assert v["positive_pairs_not_offered"] == 1
    assert v["pairs_offered"] == 8


def test_rates_and_strata(toy):
    llm, gold, _ = toy
    t = ev.per_tiab_table(llm, gold, genes=GENES)
    r = ev.rates(t)
    assert r["n_tiabs"] == 6 and r["no_match_n"] == 1 and r["unparsed_n"] == 1
    assert r["hallucinated_n"] == 1 and r["truncated_n"] == 1
    assert r["gen_tokens_max"] == 8192
    s = ev.strata(t)
    assert s["multi_gold"]["n_tiabs"] == 1 and s["cands_share_gene"]["n_tiabs"] == 1
    assert s["single_gold"]["n_tiabs"] == 3 and s["no_gold"]["n_tiabs"] == 2


def test_compare_mcnemar_and_bootstrap(tmp_path, toy):
    llm, gold, _ = toy
    t = ev.per_tiab_table(llm, gold)
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    t.to_csv(a, index=False)
    t2 = t.copy()
    t2.loc[t2["row_id"] == "b", ["exact_correct", "id_fp", "id_fn"]] = [True, 0, 0]
    t2.loc[t2["row_id"] == "b", "id_tp"] = 2
    t2.to_csv(b, index=False)
    res = ev.compare(str(a), str(b), n_boot=200)
    assert res["mcnemar"]["a_wrong_b_right"] == 1 and res["mcnemar"]["a_right_b_wrong"] == 0
    assert res["id_micro_f1_diff_a_minus_b"] < 0
    assert res["n_paired_tiabs"] == 6


def test_parse_helpers():
    assert ev.parse_set("G2P1;G2P2") == {"G2P1", "G2P2"}
    assert ev.parse_set("NO MATCH") == set() and ev.parse_set(None) == set()
    assert ev.parse_set(float("nan")) == set()
    assert ev.gold_set("") == set() and ev.gold_set("G2P1") == {"G2P1"}
    lo, hi = common.wilson_ci(9, 10)
    assert 0.55 < lo < 0.9 < hi <= 1.0


def test_prf_empty_convention():
    assert common.prf(0, 0, 0) == (0.0, 0.0, 0.0)
    assert all(math.isnan(x) for x in common.prf(0, 0, 0, empty=float("nan")))
    p, r, f = common.prf(2, 2, 0)
    assert (p, r, round(f, 4)) == (0.5, 1.0, 0.6667)


def test_id_parsers_differ_on_decorated_answers():
    decorated = "G2P00001 (EFTUD2); G2P00002."
    assert common.g2p_ids_regex(decorated) == {"G2P00001", "G2P00002"}
    assert common.g2p_ids_split(decorated) == {"G2P00001 (EFTUD2)", "G2P00002."}
    assert common.candidate_ids_from_row(None) == []
    assert common.candidate_ids_from_row(["G2P00003", "x"]) == ["G2P00003"]
