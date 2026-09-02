"""Unit tests for litdd/evaluation/llm_adjudication_eval.py on hand-checked synthetic frames.

Every metric view is exercised on a 6-TIAB toy set where the counts can be verified by hand,
including the multi-disease cases (several gold ids per TIAB, several ids in the answer).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from litdd.evaluation import llm_adjudication_eval as ev  # noqa: E402


def cand(*ids_scores):
    return [{"label": f"{i} - {g} - x", "score": s} for i, g, s in ids_scores]


@pytest.fixture()
def toy():
    # row_id, gold, answer, candidates(id, gene, score), bert
    gold = pd.DataFrame([
        {"row_id": "a", "pmid": "1", "true_g2p_ids": "G2P1", "n_gold": 1, "genereviews": False, "bert_predict": 1},
        {"row_id": "b", "pmid": "2", "true_g2p_ids": "G2P1;G2P2", "n_gold": 2, "genereviews": False, "bert_predict": 1},
        {"row_id": "c", "pmid": "3", "true_g2p_ids": "", "n_gold": 0, "genereviews": False, "bert_predict": 1},
        {"row_id": "d", "pmid": "4", "true_g2p_ids": "G2P3", "n_gold": 1, "genereviews": False, "bert_predict": 0},
        {"row_id": "e", "pmid": "5", "true_g2p_ids": "", "n_gold": 0, "genereviews": False, "bert_predict": 0},
        {"row_id": "f", "pmid": "6", "true_g2p_ids": "G2P5", "n_gold": 1, "genereviews": False, "bert_predict": 1},
    ])
    llm = pd.DataFrame([
        # a: exact hit, gated in
        {"row_id": "a", "llm_dis_map": "G2P1", "top5_cross": cand(("G2P1", "A", 0.99), ("G2P9", "B", 0.1)),
         "generated_text": "x", "answer_format_valid": True, "answer_uncertain": False,
         "answer_ids_in_candidates": True, "finish_reason": "stop", "gen_tokens": 100, "prompt_tokens": 1000},
        # b: multi-gold, answer has one right + one wrong id; both candidates share gene A
        {"row_id": "b", "llm_dis_map": "G2P1;G2P9", "top5_cross": cand(("G2P1", "A", 0.95), ("G2P2", "A", 0.5), ("G2P9", "B", 0.93)),
         "generated_text": "x", "answer_format_valid": True, "answer_uncertain": False,
         "answer_ids_in_candidates": True, "finish_reason": "stop", "gen_tokens": 300, "prompt_tokens": 1000},
        # c: no gold, NO MATCH -> correct
        {"row_id": "c", "llm_dis_map": "NO MATCH", "top5_cross": cand(("G2P7", "C", 0.2)),
         "generated_text": "x", "answer_format_valid": True, "answer_uncertain": False,
         "answer_ids_in_candidates": None, "finish_reason": "stop", "gen_tokens": 50, "prompt_tokens": 1000},
        # d: correct id but screen-negative; score below the gate
        {"row_id": "d", "llm_dis_map": "G2P3", "top5_cross": cand(("G2P3", "D", 0.4)),
         "generated_text": "x", "answer_format_valid": True, "answer_uncertain": False,
         "answer_ids_in_candidates": True, "finish_reason": "stop", "gen_tokens": 120, "prompt_tokens": 1000},
        # e: no gold, hallucinated id, truncated
        {"row_id": "e", "llm_dis_map": "G2P8", "top5_cross": cand(("G2P4", "E", 0.95)),
         "generated_text": "x", "answer_format_valid": True, "answer_uncertain": False,
         "answer_ids_in_candidates": False, "finish_reason": "length", "gen_tokens": 8192, "prompt_tokens": 1000},
        # f: unparsed
        {"row_id": "f", "llm_dis_map": None, "top5_cross": cand(("G2P5", "F", 0.97)),
         "generated_text": "x", "answer_format_valid": False, "answer_uncertain": False,
         "answer_ids_in_candidates": None, "finish_reason": "stop", "gen_tokens": 10, "prompt_tokens": 1000},
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
        {"row_id": "f", "g2p_id": "G2P6", "label": 1, "in_candidates": False},  # retriever miss
    ])
    return llm, gold, pairs


def test_per_tiab_table_sets_and_flags(toy):
    llm, gold, _ = toy
    t = ev.per_tiab_table(llm, gold, 0.9).set_index("row_id")
    assert bool(t.loc["a", "exact_correct"]) and not bool(t.loc["b", "exact_correct"])
    assert (t.loc["b", "id_tp"], t.loc["b", "id_fp"], t.loc["b", "id_fn"]) == (1, 1, 1)
    assert t.loc["b", "multi_gold"] and t.loc["b", "cands_share_gene"]
    assert not t.loc["a", "cands_share_gene"]
    assert t.loc["c", "exact_correct"] and t.loc["c", "no_match"]
    # gate: d's id scores 0.4 -> dropped when gated; b's wrong G2P9 (0.93) survives the gate
    assert (t.loc["d", "id_tp"], t.loc["d", "id_tp_gated"]) == (1, 0)
    assert (t.loc["b", "id_fp_gated"], t.loc["b", "id_tp_gated"]) == (1, 1)
    # screen gate: d is bert-negative
    assert t.loc["d", "id_tp_screen"] == 0 and t.loc["a", "id_tp_screen"] == 1
    assert t.loc["e", "hallucinated"] and t.loc["e", "truncated"]
    assert t.loc["f", "unparsed"] and t.loc["f", "format_invalid"]


def test_id_micro_counts_multi_disease(toy):
    llm, gold, _ = toy
    t = ev.per_tiab_table(llm, gold, 0.9)
    v = ev.view_id_micro(t)
    # tp: a(1) b(1) d(1) = 3 ; fp: b(1) e(1) = 2 ; fn: b(1) f(1) = 2
    assert (v["tp"], v["fp"], v["fn"]) == (3, 2, 2)
    assert v["precision"] == 0.6 and v["recall"] == 0.6
    lo, hi = v["precision_ci95"]
    assert lo < 0.6 < hi
    # gated: d's id (0.4) and e's hallucinated id (no score) drop out -> tp a,b ; fp b ; fn b,d,f
    g = ev.view_id_micro(t, "_gated")
    assert (g["tp"], g["fp"], g["fn"]) == (2, 1, 3)
    s = ev.view_id_micro(t, "_screen")  # d and e are bert-negative anyway
    assert (s["tp"], s["fp"], s["fn"]) == (2, 1, 3)


def test_tiab_exact_view(toy):
    llm, gold, _ = toy
    t = ev.per_tiab_table(llm, gold, 0.9)
    v = ev.view_tiab_exact(t)
    # exact-correct: a, c, d. tp (gold & correct): a, d = 2 ; fp (pred & wrong): b, e = 2 ;
    # fn (gold & wrong): b, f = 2 ; tn: c
    assert (v["tp"], v["fp"], v["fn"], v["tn"]) == (2, 2, 2, 1)
    assert v["accuracy"] == 0.5


def test_pair_level_view_counts_retriever_misses(toy):
    llm, gold, pairs = toy
    t = ev.per_tiab_table(llm, gold, 0.9)
    v = ev.view_pair_level(t, pairs)
    # offered pairs: a:G2P1(1,pred1) tp; a:G2P9(0,pred0) tn; b:G2P1 tp; b:G2P2 fn; c:G2P7 tn;
    # d:G2P3 tp; e:G2P4 tn (pred was G2P8, not this pair); f:G2P5 fn
    assert (v["tp"], v["fp"], v["fn"], v["tn"]) == (3, 0, 2, 3)
    assert v["positive_pairs_not_offered"] == 1
    assert v["pairs_offered_by_retriever"] == 8


def test_paper_legacy_view_reproduces_phantom_negative_rows(toy):
    llm, gold, pairs = toy
    t = ev.per_tiab_table(llm, gold, 0.9)
    v = ev.view_paper_legacy(t, pairs, 0.9)
    # a: pos row: bert1, cross(0.99>0.9)=1, set match -> TP ; a has a neg pair -> phantom
    #    neg row: bert1, not NO MATCH, cross=1 -> FP
    # b: pos row: sets differ -> FN (no neg pair -> no phantom)
    # c: neg row: bert1 & NO MATCH -> TN
    # d: pos row: bert0 -> FN
    # e: neg row: bert0 -> TN
    # f: pos row: pred empty != gold -> FN
    assert (v["tp"], v["fp"], v["fn"], v["tn"]) == (1, 1, 3, 2)


def test_rates_and_strata(toy):
    llm, gold, _ = toy
    t = ev.per_tiab_table(llm, gold, 0.9)
    r = ev.rates(t)
    assert r["n_tiabs"] == 6 and r["no_match_n"] == 1 and r["unparsed_n"] == 1
    assert r["hallucinated_n"] == 1 and r["truncated_n"] == 1
    assert r["gen_tokens_max"] == 8192
    s = ev.strata(t)
    assert s["multi_gold"]["n_tiabs"] == 1 and s["cands_share_gene"]["n_tiabs"] == 1
    assert s["single_gold"]["n_tiabs"] == 3 and s["no_gold"]["n_tiabs"] == 2


def test_compare_mcnemar_and_bootstrap(tmp_path, toy):
    llm, gold, _ = toy
    t = ev.per_tiab_table(llm, gold, 0.9)
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
    lo, hi = ev.wilson(9, 10)
    assert 0.55 < lo < 0.9 < hi <= 1.0
