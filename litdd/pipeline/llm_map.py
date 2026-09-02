"""LLM adjudication stage: map each screened, cross-encoded TIAB to G2P entries.

Reads the cross-encoder shards (``tiab`` + ``top5_cross``), renders the adjudication prompt
for each row, runs the model under vLLM and writes ``{shard}__llm.parquet`` with the parsed
answer in ``llm_dis_map`` (``G2Pxxxxx``, ``G2Pa;G2Pb`` or ``NO MATCH``). Downstream,
``final_data_clean.py`` reads exactly ``pmid``, ``llm_dis_map`` and ``top5_cross``; the last
is passed through untouched because it carries the scores for the 0.9 gate.

The prompt lives in ``prompts/original_paper.txt`` (the verbatim text reported in the
manuscript) and is rendered through the model's chat template, so instruct/reasoning models
see a proper user turn rather than a bare completion string. Reasoning effort, decoding and
engine limits are explicit CLI arguments and are recorded per shard in ``run_meta.json``.

torch and vllm are imported lazily inside ``run_llm_over_cross_shards`` so the deterministic
helpers (prompt building, answer parsing, sharding) import and unit-test without the GPU stack.
"""
from __future__ import annotations

import argparse
import functools
import gc
import glob
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd

PROMPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
DEFAULT_PROMPT_FILE = os.path.join(PROMPT_DIR, "original_paper.txt")

G2P_ID_RE = re.compile(r"G2P\d+")
NO_MATCH = "NO MATCH"


# --------------------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=8)
def load_prompt_template(path: str = DEFAULT_PROMPT_FILE) -> str:
    """Read a prompt template. Placeholders: {n}, {plural}, {tiab}, {candidate_lines}."""
    with open(path, encoding="utf-8") as f:
        template = f.read()
    for key in ("{n}", "{tiab}", "{candidate_lines}"):
        if key not in template:
            raise ValueError(f"prompt template {path} lacks the {key} placeholder")
    return template


def gene_of(candidate: str) -> str:
    """Gene symbol of a candidate: 2nd ' - ' field of a flat thread, or the 'Gene Symbol:'
    line of a contextualised block; '' when neither is present."""
    m = re.search(r"^Gene Symbol:\s*(.+)$", str(candidate), flags=re.MULTILINE)
    if m:
        return m.group(1).strip()
    parts = str(candidate).split(" - ")
    return parts[1].strip() if len(parts) > 1 else ""


def render_candidates(candidate_lines, layout: str = "flat") -> str:
    """Number the candidates; ``layout='by_gene'`` groups them under one header per gene.

    With the gene gate every entry of a mentioned gene is a candidate, so an allelic series
    arrives as several near-identical lines interleaved with other genes' entries. Grouping
    them makes the structure explicit -- 'these N entries are alternatives for the same gene;
    pick among them' -- without changing the numbering the answer refers to.
    """
    candidate_lines = list(candidate_lines)
    if layout == "flat":
        return "\n".join(f"{i + 1}) {c}" for i, c in enumerate(candidate_lines))
    if layout != "by_gene":
        raise ValueError(f"unknown candidate layout {layout!r}")
    groups: dict[str, list[tuple[int, str]]] = {}
    for i, c in enumerate(candidate_lines):
        groups.setdefault(gene_of(c) or "(gene not stated)", []).append((i, c))
    out = []
    for gene, items in groups.items():   # insertion order = cross-encoder score order
        n = len(items)
        out.append(f"Gene {gene} — {n} candidate {'entry' if n == 1 else 'entries (alternative disorders of this gene; choose among them)'}:")
        out += [f"{i + 1}) {c}" for i, c in items]
    return "\n".join(out)


def build_llm_prompt(tiab, candidate_lines, template_path: str = DEFAULT_PROMPT_FILE,
                     layout: str = "flat"):
    """Render the adjudication prompt for one TIAB and its candidate threads.

    The candidate count is data-driven, not fixed at 5: with the gene-mention filter moved
    ahead of the cross-encoder, a TIAB gets as many candidates as its mentioned genes
    support, which may be fewer or more than five. The prompt therefore states the actual
    number rather than hard-coding "5", and candidates are numbered so that multi-line
    (contextualised) threads have unambiguous boundaries.

    Raises on an empty candidate list. An empty list would render a prompt with no
    candidates at all, and the model would dutifully answer NO MATCH -- indistinguishable
    from a real negative. Silently mapping a whole corpus to NO MATCH is the failure mode
    this guard exists to prevent, so callers must filter or handle empties explicitly.
    """
    candidate_lines = list(candidate_lines)
    n = len(candidate_lines)
    if n == 0:
        raise ValueError(
            "build_llm_prompt called with no candidate threads. This would produce a "
            "prompt containing zero candidates and an unconditional 'NO MATCH' answer. "
            "Filter these rows out upstream, or record them as no-candidate rather than "
            "sending them to the LLM."
        )
    plural = "thread" if n == 1 else "threads"
    numbered = render_candidates(candidate_lines, layout)
    return load_prompt_template(template_path).format(
        n=n, plural=plural, tiab=tiab, candidate_lines=numbered
    )


# --------------------------------------------------------------------------------------
# Answer parsing
# --------------------------------------------------------------------------------------
def extract_last_answer(text):
    """Text after the LAST ``ANSWER:`` in the generation, or None.

    Taking the last occurrence tolerates reasoning traces that quote the schema before the
    final line (DeepSeek-R1 ``<think>`` blocks; GPT-OSS harmony output where the analysis
    channel and the final channel are concatenated as ``...assistantfinalANSWER: ...``).
    """
    # Locate every marker first (a greedy `(.*)` would swallow a later marker on the same
    # line, which is exactly the harmony ``...assistantfinalANSWER:`` case).
    marks = list(re.finditer(r"ANSWER:\s*", text or "", flags=re.IGNORECASE))
    if not marks:
        return None
    return (text[marks[-1].end():].split("\n", 1)[0]).strip()


def candidate_ids(candidate_lines) -> list[str]:
    """The G2P id at the head of each candidate thread (flat or contextualised)."""
    ids = []
    for c in candidate_lines:
        m = G2P_ID_RE.search(str(c))
        ids.append(m.group(0) if m else None)
    return ids


ROLES_MAPPED = ("causal", "co-causal")


def extract_json_answer(text) -> dict | None:
    """The LAST well-formed JSON object in the generation (harmony's final channel comes
    last), or None. Tolerates ```json fences and text before/after the object."""
    if not text:
        return None
    s = str(text)
    end = s.rfind("}")
    while end != -1:
        depth = 0
        for start in range(end, -1, -1):
            if s[start] == "}":
                depth += 1
            elif s[start] == "{":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(s[start:end + 1])
                        if isinstance(obj, dict) and "genes" in obj:
                            return obj
                    except json.JSONDecodeError:
                        pass
                    break
        end = s.rfind("}", 0, end)
    return None


def parse_json_answer(obj: dict | None, allowed_ids=None) -> dict:
    """Map the structured per-gene answer onto the llm_dis_map contract.

    Entries of genes whose role is causal / co-causal form the mapping; every other role
    contributes nothing. The 'answer' field is cross-checked and the per-gene roles and
    confidences are kept so the adjudicator's reasons become auditable columns.
    """
    out = {"llm_dis_map": None, "answer_format_valid": False, "answer_uncertain": False,
           "answer_ids_in_candidates": None, "llm_roles": None, "llm_confidence_min": None,
           "json_answer_consistent": None}
    if not obj:
        return out
    genes = obj.get("genes") or []
    ids, roles, confs = [], [], []
    ok = isinstance(genes, list)
    for g in genes if ok else []:
        if not isinstance(g, dict):
            ok = False
            continue
        role = str(g.get("role", "")).strip().lower()
        roles.append({"gene": g.get("gene"), "role": role, "confidence": g.get("confidence")})
        if role in ROLES_MAPPED:
            ents = g.get("entries") or ([g["entry"]] if g.get("entry") else [])
            for e in ents:
                m = G2P_ID_RE.search(str(e))
                if m and m.group(0) not in ids:
                    ids.append(m.group(0))
            try:
                confs.append(float(g.get("confidence")))
            except (TypeError, ValueError):
                pass
    out["llm_roles"] = json.dumps(roles)
    out["llm_confidence_min"] = min(confs) if confs else None
    out["llm_dis_map"] = ";".join(ids) if ids else NO_MATCH
    out["answer_format_valid"] = ok
    ans = obj.get("answer")
    if ans is not None:
        ans_ids = set(G2P_ID_RE.findall(str(ans)))
        out["json_answer_consistent"] = (ans_ids == set(ids)) if ids else (str(ans).strip().upper() == NO_MATCH)
    if allowed_ids is not None and ids:
        allowed = {a for a in allowed_ids if a}
        out["answer_ids_in_candidates"] = all(i in allowed for i in ids)
    return out


def parse_answer(raw_answer, allowed_ids=None) -> dict:
    """Normalise the text after ``ANSWER:`` into the ``llm_dis_map`` contract.

    Returns a dict with:
      llm_dis_map            "G2Pxxxxx", "G2Pa;G2Pb", "NO MATCH", or None (no parseable answer)
      answer_format_valid    True when the answer is exactly the schema (ids ; ids, NO MATCH)
      answer_uncertain       True when the model said UNCERTAIN (mapped to NO MATCH)
      answer_ids_in_candidates
                             True when every returned id is one of the offered candidates
                             (None when there are no ids or no candidate list was given)

    IDs are extracted with ``G2P\\d+`` so decorations ("G2P01236 (EFTUD2)", markdown, a
    trailing full stop) do not turn a correct answer into a "hallucination" downstream.
    Hallucinated ids are NOT removed here -- ``final_data_clean.py`` drops them against the
    panel -- they are only flagged so the rate is measurable.
    """
    out = {
        "llm_dis_map": None,
        "answer_format_valid": False,
        "answer_uncertain": False,
        "answer_ids_in_candidates": None,
    }
    if raw_answer is None:
        return out
    text = str(raw_answer).strip().strip("`*\"' ").rstrip(".").strip()
    upper = text.upper()
    if "UNCERTAIN" in upper and not G2P_ID_RE.search(text):
        out.update(llm_dis_map=NO_MATCH, answer_uncertain=True,
                   answer_format_valid=upper == "UNCERTAIN")
        return out
    if "NO MATCH" in upper and not G2P_ID_RE.search(text):
        out.update(llm_dis_map=NO_MATCH, answer_format_valid=upper == NO_MATCH)
        return out
    ids = []
    for m in G2P_ID_RE.findall(text):
        if m not in ids:
            ids.append(m)
    if not ids:
        return out
    out["llm_dis_map"] = ";".join(ids)
    out["answer_format_valid"] = re.fullmatch(r"G2P\d+(\s*;\s*G2P\d+)*", text) is not None
    if allowed_ids is not None:
        allowed = {a for a in allowed_ids if a}
        out["answer_ids_in_candidates"] = all(i in allowed for i in ids)
    return out


# --------------------------------------------------------------------------------------
# Candidate rendering
# --------------------------------------------------------------------------------------
def to_labels(x, max_candidates=None, min_score=None, show_scores=False):
    """Normalise a top-k cell into a list of candidate label strings.

    max_candidates=None keeps every candidate, which is what the data-driven
    configuration wants: the number of candidates follows from the genes actually
    mentioned in the TIAB, so it is not fixed at 5.

    min_score drops candidates whose cross-encoder score is below the gate BEFORE the
    LLM sees them (deployment order: gene gate -> cross-encoder on every entry of the
    detected genes -> score gate -> LLM). Items without a score are kept.
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []

    def item_to_pair(item):
        """-> (label, score or None) or None."""
        if isinstance(item, dict):
            lab = str(item.get("label", "")).strip()
            return (lab, item.get("score")) if lab else None
        if isinstance(item, (list, tuple)) and len(item) >= 1:
            lab = str(item[0]).strip()
            sc = item[1] if len(item) > 1 else None
            return (lab, sc) if lab else None
        try:
            import pyarrow as pa
            if isinstance(item, pa.Scalar):
                return item_to_pair(item.as_py())
        except Exception:  # noqa: BLE001
            pass
        if isinstance(item, str):
            return (item.strip(), None) if item.strip() else None
        return None

    if isinstance(x, (list, tuple, np.ndarray)):
        labels = []
        for it in (x.tolist() if isinstance(x, np.ndarray) else x):
            pair = item_to_pair(it)
            if not pair:
                continue
            lab, sc = pair
            if min_score is not None and sc is not None:
                try:
                    if float(sc) < min_score:
                        continue
                except (TypeError, ValueError):
                    pass
            if show_scores and sc is not None:
                try:
                    lab = f"{lab} [retrieval score {float(sc):.2f}]"
                except (TypeError, ValueError):
                    pass
            labels.append(lab)
        return labels if max_candidates is None else labels[:max_candidates]

    if isinstance(x, str):
        import ast
        obj = None
        try:
            obj = json.loads(x)
        except Exception:  # noqa: BLE001
            try:
                obj = ast.literal_eval(x)
            except Exception:  # noqa: BLE001
                return []
        return to_labels(obj, max_candidates, min_score)

    try:
        import pyarrow as pa
        if isinstance(x, pa.Scalar):
            return to_labels(x.as_py(), max_candidates, min_score)
    except Exception:  # noqa: BLE001
        pass
    return []


SKIPPED_TEXT = "[skipped: no candidate passed the score gate]"
SKIPPED_TOO_LONG_TEXT = "[skipped: prompt exceeds the model context even undecorated]"


def load_context_threads(path: str) -> dict[str, str]:
    """``{g2p_id: contextualised multi-line thread}`` from the offline-built JSON.

    Lines whose value is the literal ``None`` (an empty enrichment field) are dropped:
    they carry no information and cost tokens on every candidate.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        if k.startswith("__"):
            continue
        lines = [ln for ln in str(v).splitlines() if not ln.rstrip().endswith(": None")]
        out[k] = "\n".join(lines).strip()
    return out


def load_hpo_terms(path: str) -> dict[str, list[dict]]:
    """``{g2p_id: [{id, name, freq: [...], pmids: [...]}, ...]}`` from build_hpo_terms.py."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("__")}


def hpo_decorate(labels, hpo: dict[str, list[dict]], pmid=None, max_terms=None,
                 multi_only=False):
    """Append each candidate's amalgamated HPO phenotype list (name + frequency).

    ``pmid`` is the abstract being adjudicated: terms whose ONLY provenance is that very
    PMID are dropped (the phenotype.hpoa row was curated FROM this paper — showing it back
    to the model would leak the answer into the prompt).

    ``max_terms`` caps the list per candidate (terms with a curated frequency first) with an
    explicit "(+n more)" marker — the fallback for abstracts whose full decoration would not
    fit the model context.
    """
    pmid = str(pmid or "").removeprefix("pmid")
    if multi_only:
        counts = {}
        for lab in labels:
            g = gene_of(lab)
            counts[g] = counts.get(g, 0) + 1
    out = []
    for lab in labels:
        m = G2P_ID_RE.search(str(lab))
        terms = hpo.get(m.group(0)) if m else None
        if multi_only and counts.get(gene_of(lab), 0) < 2:
            terms = None
        if not terms:
            out.append(lab)
            continue
        kept = [t for t in terms
                if not (pmid and t.get("pmids") and all(p == pmid for p in t["pmids"]))]
        extra = 0
        if max_terms is not None and len(kept) > max_terms:
            kept = sorted(kept, key=lambda t: not t.get("freq"))[:max_terms]
            extra = len(terms) - max_terms
        parts = [f"{t['name']} ({'; '.join(t['freq'])})" if t.get("freq") else t["name"]
                 for t in kept]
        if extra:
            parts.append(f"(+{extra} more)")
        out.append(f"{lab}\n    Phenotypes (HPO): {'; '.join(parts)}" if parts else lab)
    return out


def contextualise(labels, context: dict[str, str], missing_counter: dict | None = None):
    """Swap each flat thread for its contextualised block, falling back to the flat text.

    A missing id is a panel-version mismatch (the JSON was built from a different G2P
    export than the candidates). It is counted rather than raised so a handful of retired
    entries do not kill a corpus run, but the count is reported in run_meta.json.
    """
    out = []
    for lab in labels:
        m = G2P_ID_RE.search(lab)
        gid = m.group(0) if m else None
        if gid in context:
            out.append(context[gid])
        else:
            out.append(lab)
            if missing_counter is not None:
                missing_counter["missing"] = missing_counter.get("missing", 0) + 1
    return out


# --------------------------------------------------------------------------------------
# Sharding
# --------------------------------------------------------------------------------------
def batched_indices(start, end, batch_size):
    i = start
    while i < end:
        j = min(i + batch_size, end)
        yield i, j
        i = j


def select_shards_for_worker(all_paths, shard_index, num_shards):
    """File-level work split. Prefer `row_slice_for_worker` when files < workers.

    Kept because it is correct when there are at least as many shard files as workers, and
    because existing manifests call it. The deployed run had 4 files and requested 8 workers,
    so workers 4-7 received an empty list, printed "Found 0 parquet shard(s)" and exited --
    half of an 8x A100 allocation sat idle for six days. `run_llm_over_cross_shards` now
    row-shards instead, which is correct for any file:worker ratio.
    """
    if shard_index is None or num_shards is None:
        return all_paths
    return [p for i, p in enumerate(all_paths) if (i % num_shards) == shard_index]


def row_slice_for_worker(n_rows, shard_index, num_shards):
    """Row indices this worker owns, striped so every worker gets work from every file.

    Striping rather than contiguous blocks keeps the workers balanced even when a file's rows
    vary in prompt length, and means adding a worker does not reshuffle the others' spans.
    """
    if shard_index is None or num_shards is None or num_shards <= 1:
        return list(range(n_rows))
    return list(range(shard_index, n_rows, num_shards))


# --------------------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------------------
def _git_sha() -> str | None:
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        # Pods run as root over a user-owned checkout; without safe.directory git refuses.
        return subprocess.run(["git", "-c", "safe.directory=*", "-C", here, "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _versions() -> dict:
    v = {"python": sys.version.split()[0]}
    for mod in ("vllm", "torch", "transformers"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            v[mod] = None
    return v


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------
def run_llm_over_cross_shards(
    shards_dir,
    llm_model,
    out_dir=None,
    temperature=0.0,
    top_p=1.0,
    max_tokens=8192,
    shard_index=None,
    num_shards=None,
    save_every=1000,
    tensor_parallel_size=None,
    max_candidates=None,
    min_score=None,
    max_num_seqs=512,
    max_model_len=16384,
    gpu_memory_utilization=0.90,
    dtype="auto",
    seed=0,
    reasoning_effort="medium",
    use_chat_template=True,
    prompt_file=DEFAULT_PROMPT_FILE,
    threads="vanilla",
    show_scores=False,
    hpo_json=None,
    hpo_multi_only=False,
    context_json=None,
    limit=None,
    candidate_layout="flat",
    output_format="answer",
):
    """
    - Reads *.parquet from shards_dir
    - Builds prompts from 'tiab' and 'top5_cross' (optionally swapping in contextualised threads)
    - Runs vLLM once per checkpoint window (continuous batching), resumable per shard
    - Writes {shard}[_w{shard_index}]__llm.parquet = input columns +
        topk_cross_lgmde / top_5_cross_lgmde  (candidate text the LLM saw)
        llm_prompt, generated_text, llm_answer_raw, llm_dis_map,
        answer_format_valid, answer_uncertain, answer_ids_in_candidates,
        finish_reason, prompt_tokens, gen_tokens
      and {shard}[_w{shard_index}]__llm.run_meta.json with settings, versions and throughput.
    """
    import torch
    from vllm import LLM, SamplingParams

    out_dir = out_dir or shards_dir
    os.makedirs(out_dir, exist_ok=True)

    context = None
    context_missing = {}
    hpo_terms = load_hpo_terms(hpo_json) if hpo_json else None
    if hpo_terms is not None:
        print(f"Loaded HPO terms for {len(hpo_terms)} G2P entries from {hpo_json}")
    if threads == "context":
        if not context_json:
            raise ValueError("--threads context requires --context_json")
        context = load_context_threads(context_json)
        print(f"Loaded {len(context)} contextualised threads from {context_json}")

    sampling_params = SamplingParams(temperature=temperature, top_p=top_p,
                                     max_tokens=max_tokens, seed=seed)
    # The prompt is a ~5,000-character fixed rubric plus a short per-record suffix, so the
    # shared prefix dominates. Without prefix caching that rubric is re-prefilled once per
    # record -- ~10^9 redundant prefill tokens over a full corpus.
    llm_kwargs = {
        "enable_prefix_caching": True,
        "max_num_seqs": max_num_seqs,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "seed": seed,
    }
    if dtype and dtype != "auto":
        llm_kwargs["dtype"] = dtype
    if tensor_parallel_size is not None:
        llm_kwargs["tensor_parallel_size"] = int(tensor_parallel_size)
    chat_kwargs = {}
    if reasoning_effort:
        # GPT-OSS reads this from the chat template ("Reasoning: medium"); models whose
        # template does not use it ignore the variable.
        chat_kwargs["reasoning_effort"] = reasoning_effort

    t_engine = time.time()
    llm = LLM(model=llm_model, **llm_kwargs)
    print(f"Engine up in {time.time() - t_engine:.0f}s")

    settings = {
        "stage": "llm_map",
        "model": llm_model,
        "prompt_file": os.path.abspath(prompt_file),
        "use_chat_template": use_chat_template,
        "reasoning_effort": reasoning_effort if use_chat_template else None,
        "threads": threads,
        "candidate_layout": candidate_layout,
        "output_format": output_format,
        "show_scores": show_scores,
        "hpo_json": os.path.abspath(hpo_json) if hpo_json else None,
        "hpo_multi_only": hpo_multi_only,
        "context_json": os.path.abspath(context_json) if context_json else None,
        "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens, "seed": seed,
        "max_model_len": max_model_len, "max_num_seqs": max_num_seqs,
        "gpu_memory_utilization": gpu_memory_utilization, "dtype": dtype,
        "tensor_parallel_size": tensor_parallel_size, "max_candidates": max_candidates,
        "min_score": min_score,
        "shard_index": shard_index, "num_shards": num_shards,
        "git_sha": _git_sha(), "image": os.environ.get("LITDD_IMAGE"),
        "versions": _versions(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    shard_paths = sorted(glob.glob(os.path.join(shards_dir, "*.parquet")))
    print(f"Found {len(shard_paths)} parquet shard(s) for this worker.")

    extra_cols = ["llm_answer_raw", "llm_dis_map", "answer_format_valid", "answer_uncertain",
                  "llm_roles", "llm_confidence_min", "json_answer_consistent",
                  "answer_ids_in_candidates", "finish_reason", "prompt_tokens", "gen_tokens"]

    for shard_path in shard_paths:
        print(f"Processing shard: {os.path.basename(shard_path)}")
        t_shard = time.time()
        df = pd.read_parquet(shard_path)
        if num_shards and num_shards > 1:
            keep = row_slice_for_worker(len(df), shard_index, num_shards)
            df = df.iloc[keep].reset_index(drop=True)
            print(f"[shard {shard_index}/{num_shards}] rows for this worker: {len(df)}")
        if limit:
            df = df.iloc[:limit].reset_index(drop=True)
        if df.empty:
            print("  (no rows for this worker in this file)")
            continue

        n_uncapped = df["top5_cross"].apply(lambda x: len(to_labels(x, None)))
        n_gated = df["top5_cross"].apply(lambda x: len(to_labels(x, None, min_score)))
        flat = df["top5_cross"].apply(lambda x: to_labels(x, max_candidates, min_score,
                                                          show_scores=show_scores))
        capped_rows = int((n_gated > max_candidates).sum()) if max_candidates else 0
        if capped_rows:
            print(f"[CAP] {capped_rows} rows had more than {max_candidates} candidates; "
                  f"showing the top {max_candidates} by cross-encoder score")
        # Rows with no candidate left after the score gate never reach the model: they are
        # recorded as NO MATCH with finish_reason="skipped" so the rate is visible downstream.
        skipped = flat.apply(len) == 0
        if min_score is not None:
            print(f"[GATE] score >= {min_score}: {int((n_uncapped - n_gated).sum())} of "
                  f"{int(n_uncapped.sum())} candidates removed; {int(skipped.sum())} of {len(df)} "
                  f"rows left with no candidate (-> NO MATCH, not sent to the LLM)")
        elif skipped.any():
            print(f"[WARN] {int(skipped.sum())} rows have no candidates at all (-> NO MATCH)")
        df["topk_cross_lgmde"] = flat.apply(
            lambda labs: contextualise(labs, context, context_missing) if context else labs)
        base_labels = df["topk_cross_lgmde"].tolist()
        if hpo_terms is not None:
            df["topk_cross_lgmde"] = [
                hpo_decorate(labs, hpo_terms, pmid=pm, multi_only=hpo_multi_only)
                for labs, pm in zip(base_labels, df.get("pmid", [None] * len(df)))
            ]
        # Legacy alias: existing analyses (cascade_funnel, sample_audit) read this name.
        df["top_5_cross_lgmde"] = df["topk_cross_lgmde"]
        df["llm_prompt"] = [
            build_llm_prompt(t, labs, template_path=prompt_file, layout=candidate_layout)
            if labs else None
            for t, labs in zip(df.get("tiab", pd.Series([""] * len(df))), df["topk_cross_lgmde"])
        ]
        allowed = flat.apply(candidate_ids)

        # Context guard: a prompt longer than the model context makes vLLM abort the whole
        # batch. Degrade per row: cap the HPO list, then drop the decoration, then skip the
        # row explicitly (finish_reason="too_long") -- never crash a corpus run over one
        # many-gene review abstract.
        tokenizer = llm.get_tokenizer()
        budget = max_model_len - max(1024, max_tokens // 4)
        pmids_seq = df.get("pmid", pd.Series([None] * len(df))).tolist()
        rows_trimmed = rows_undecorated = 0
        too_long_rows = set()
        new_prompts = df["llm_prompt"].tolist()
        new_cands = df["topk_cross_lgmde"].tolist()
        for i, prompt in enumerate(new_prompts):
            if prompt is None or len(prompt) < budget * 2:   # cheap lower bound: >=2 chars/token
                continue
            if len(tokenizer.encode(prompt)) <= budget:
                continue
            labs = base_labels[i]
            candidates_attempts = []
            if hpo_terms is not None:
                candidates_attempts.append(hpo_decorate(labs, hpo_terms, pmid=pmids_seq[i],
                                                        max_terms=15, multi_only=hpo_multi_only))
            candidates_attempts.append(labs)
            for attempt, cand in enumerate(candidates_attempts):
                new_prompt = build_llm_prompt(df["tiab"].iloc[i], cand,
                                              template_path=prompt_file, layout=candidate_layout)
                if len(tokenizer.encode(new_prompt)) <= budget:
                    new_prompts[i] = new_prompt
                    new_cands[i] = cand
                    if hpo_terms is not None and attempt == 0:
                        rows_trimmed += 1
                    else:
                        rows_undecorated += 1
                    break
            else:
                too_long_rows.add(i)
        df["llm_prompt"] = new_prompts
        df["topk_cross_lgmde"] = new_cands
        df["top_5_cross_lgmde"] = df["topk_cross_lgmde"]
        if rows_trimmed or rows_undecorated or too_long_rows:
            print(f"[CONTEXT] over-budget prompts: {rows_trimmed} HPO-capped, "
                  f"{rows_undecorated} undecorated, {len(too_long_rows)} skipped as too long")

        first_prompt = next((p for p in df["llm_prompt"] if p), "")
        print("Prompt preview:\n", first_prompt[-600:])
        N = len(df)
        print(f"Total rows in shard: {N}")

        base = os.path.splitext(os.path.basename(shard_path))[0]
        suffix = "" if shard_index is None else f"_w{shard_index}"
        out_parquet = os.path.join(out_dir, f"{base}{suffix}__llm.parquet")
        meta_path = os.path.join(out_dir, f"{base}{suffix}__llm.run_meta.json")

        # Resume: on a preemptible cluster a multi-hour shard will be interrupted, and the
        # previous implementation restarted every shard from row 0.
        generated_texts = [""] * N
        extras = {c: [None] * N for c in extra_cols}
        if os.path.exists(out_parquet):
            try:
                prev = pd.read_parquet(out_parquet)
                if len(prev) == N:
                    generated_texts = ["" if pd.isna(t) else str(t)
                                       for t in prev["generated_text"].tolist()]
                    for c in extra_cols:
                        if c in prev.columns:
                            extras[c] = prev[c].tolist()
                    done = sum(1 for t in generated_texts if t)
                    print(f"[RESUME] {done}/{N} rows already generated in {out_parquet}")
                else:
                    print(f"[RESUME] row-count mismatch ({len(prev)} != {N}); starting fresh")
            except Exception as e:  # noqa: BLE001 - a corrupt checkpoint must not block the run
                print(f"[RESUME] could not read {out_parquet} ({e}); starting fresh")

        def save_progress():
            df["generated_text"] = generated_texts
            for c in extra_cols:
                df[c] = extras[c]
            df.to_parquet(out_parquet, index=False)
            print(f"[PROGRESS] Saved current progress to {out_parquet}", flush=True)

        for i in too_long_rows:
            if generated_texts[i]:
                continue
            generated_texts[i] = SKIPPED_TOO_LONG_TEXT
            extras["llm_dis_map"][i] = None
            extras["finish_reason"][i] = "too_long"
            extras["prompt_tokens"][i] = 0
            extras["gen_tokens"][i] = 0
        for i in np.flatnonzero(skipped.to_numpy()):
            generated_texts[i] = SKIPPED_TEXT
            extras["llm_answer_raw"][i] = None
            extras["llm_dis_map"][i] = NO_MATCH
            extras["answer_format_valid"][i] = True
            extras["answer_uncertain"][i] = False
            extras["answer_ids_in_candidates"][i] = None
            extras["finish_reason"][i] = "skipped"
            extras["prompt_tokens"][i] = 0
            extras["gen_tokens"][i] = 0

        todo = [i for i in range(N) if not generated_texts[i]]
        if not todo:
            print("[SKIP] shard already complete")
            continue

        # One generate() call per checkpoint window, not per fixed small batch.
        #
        # The previous code called llm.generate() on 12 prompts at a time, which defeats
        # vLLM's continuous batching: each call spins the scheduler up from zero, runs at
        # concurrency 12 and drains, so the KV cache never fills. The deployed run measured
        # ~186 output tokens/s for a dense 14B model on an 80GB A100 -- roughly an order of
        # magnitude below what that hardware does. Handing vLLM the whole window lets it
        # schedule continuously; `save_every` now controls only checkpoint granularity.
        window = save_every if save_every and save_every > 0 else N
        t_gen = time.time()
        for w_start in range(0, len(todo), window):
            idx = todo[w_start:w_start + window]
            prompts = [df["llm_prompt"].iloc[i] for i in idx]
            print(f"  generating {len(prompts)} prompt(s) "
                  f"[{w_start + len(idx)}/{len(todo)} of this shard's outstanding rows]",
                  flush=True)
            if use_chat_template:
                conversations = [[{"role": "user", "content": p}] for p in prompts]
                outputs = llm.chat(conversations, sampling_params, use_tqdm=True,
                                   chat_template_kwargs=chat_kwargs or None)
            else:
                outputs = llm.generate(prompts, sampling_params)
            for i, out in zip(idx, outputs):
                comp = out.outputs[0]
                text = comp.text
                generated_texts[i] = text if text else " "  # keep non-empty so resume skips it
                if output_format == "json":
                    obj = extract_json_answer(text)
                    parsed = parse_json_answer(obj, allowed.iloc[i])
                    raw = json.dumps(obj) if obj else extract_last_answer(text)
                    if not obj:   # no JSON object: fall back to the ANSWER: line if present
                        fb = parse_answer(raw, allowed.iloc[i])
                        parsed.update({k: fb[k] for k in ("llm_dis_map", "answer_uncertain",
                                                          "answer_ids_in_candidates")})
                        parsed["answer_format_valid"] = False
                else:
                    raw = extract_last_answer(text)
                    parsed = parse_answer(raw, allowed.iloc[i])
                extras["llm_answer_raw"][i] = raw
                for k, v in parsed.items():
                    extras[k][i] = v
                extras["finish_reason"][i] = comp.finish_reason
                extras["prompt_tokens"][i] = len(out.prompt_token_ids or [])
                extras["gen_tokens"][i] = len(comp.token_ids or [])
            save_progress()
        gen_seconds = time.time() - t_gen

        n_done = len(todo)
        gen_tok = [extras["gen_tokens"][i] or 0 for i in todo]
        prm_tok = [extras["prompt_tokens"][i] or 0 for i in todo]
        meta = dict(settings)
        meta.update({
            "shard": os.path.basename(shard_path), "out_parquet": out_parquet,
            "rows_total": N, "rows_generated_this_run": n_done,
            "wall_clock_s": round(time.time() - t_shard, 1),
            "generation_s": round(gen_seconds, 1),
            "rows_per_s": round(n_done / gen_seconds, 3) if gen_seconds else None,
            "prompt_tokens_total": int(sum(prm_tok)), "gen_tokens_total": int(sum(gen_tok)),
            "gen_tokens_mean": float(np.mean(gen_tok)) if gen_tok else None,
            "gen_tokens_p95": float(np.percentile(gen_tok, 95)) if gen_tok else None,
            "gen_tokens_per_s": round(sum(gen_tok) / gen_seconds, 1) if gen_seconds else None,
            "truncated_rows": int(sum(1 for i in todo if extras["finish_reason"][i] == "length")),
            "no_match_rows": int(sum(1 for i in todo if extras["llm_dis_map"][i] == NO_MATCH)),
            "unparsed_rows": int(sum(1 for i in todo if extras["llm_dis_map"][i] is None)),
            "hallucinated_rows": int(sum(1 for i in todo
                                         if extras["answer_ids_in_candidates"][i] is False)),
            "context_threads_missing": context_missing.get("missing", 0),
            "candidates_per_row_mean": float(n_uncapped.mean()),
            "candidates_per_row_max": int(n_uncapped.max()),
            "rows_capped_by_max_candidates": capped_rows,
            "candidates_removed_by_min_score": int((n_uncapped - n_gated).sum()),
            "rows_skipped_no_candidates": int(skipped.sum()),
            "rows_hpo_capped": rows_trimmed, "rows_undecorated": rows_undecorated,
            "rows_skipped_too_long": len(too_long_rows),
            # (no peak-memory field: vLLM v1 runs the engine in a child process, so the
            # driver's torch.cuda counters read 0; gpu_memory_utilization is the budget.)
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[DONE] Shard completed: {os.path.basename(shard_path)} "
              f"({n_done} rows, {meta['rows_per_s']} rows/s, "
              f"{meta['gen_tokens_mean']:.0f} mean gen tokens, "
              f"{meta['truncated_rows']} truncated, {meta['unparsed_rows']} unparsed)")

        torch.cuda.empty_cache()
        gc.collect()

    print("All shards processed.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--shards_dir", required=True, type=str)
    p.add_argument("--llm_model", required=True, type=str,
                   help="HF id or local path; deployed: openai/gpt-oss-20b")
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=None,
                   help="DEPRECATED, ignored: generation is one call per --save_every window.")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Mapping is treated as deterministic; default 0.0.")
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max_tokens", type=int, default=8192,
                   help="Generation budget incl. the reasoning trace; rows that hit it are "
                        "flagged finish_reason=length and counted in run_meta.json.")
    p.add_argument("--max_model_len", type=int, default=16384)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--dtype", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reasoning_effort", type=str, default="medium",
                   choices=["low", "medium", "high", "none"],
                   help="Passed to the chat template (GPT-OSS). 'none' omits it.")
    p.add_argument("--no_chat_template", action="store_true",
                   help="Send the prompt as a raw completion string (the pre-revision "
                        "behaviour, kept so the DeepSeek deployment can be reproduced).")
    p.add_argument("--prompt_file", type=str, default=DEFAULT_PROMPT_FILE)
    p.add_argument("--threads", type=str, default="vanilla", choices=["vanilla", "context"],
                   help="Candidate representation: the flat 15-field thread the "
                        "cross-encoder scored, or the contextualised multi-line block "
                        "(needs --context_json built from the SAME G2P export).")
    p.add_argument("--context_json", type=str, default=None)
    p.add_argument("--hpo_multi_only", action="store_true",
                   help="Decorate with HPO terms only the candidates whose gene has more than "
                        "one entry among this abstract's candidates (the allelic-series "
                        "disambiguation case), leaving single-entry candidates unchanged.")
    p.add_argument("--hpo_json", type=str, default=None,
                   help="build_hpo_terms.py output: append each candidate's amalgamated HPO "
                        "phenotypes (name + frequency) to its line; terms curated solely from "
                        "the abstract's own PMID are dropped (leakage guard).")
    p.add_argument("--show_scores", action="store_true",
                   help="Append each candidate's cross-encoder score to its line "
                        "('[retrieval score 0.97]') so the LLM can use retrieval strength "
                        "as evidence, especially between siblings of one gene.")
    p.add_argument("--output_format", type=str, default="answer", choices=["answer", "json"],
                   help="answer: the 'ANSWER: ids' line (original rubric). json: the structured "
                        "per-gene object (role / entries / confidence) of prompts/"
                        "original_paper_json.txt -- use with --prompt_file pointing at it; "
                        "llm_dis_map = entries of causal and co-causal genes.")
    p.add_argument("--candidate_layout", type=str, default="flat", choices=["flat", "by_gene"],
                   help="How candidates are listed: one numbered line each (flat, the paper's "
                        "layout) or grouped under a header per gene so an allelic series is "
                        "visibly one choice (by_gene). Numbering is unchanged.")
    p.add_argument("--shard_index", type=int, default=None)
    p.add_argument("--num_shards", type=int, default=None)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--tensor_parallel_size", type=int, default=None)
    p.add_argument("--max_num_seqs", type=int, default=512,
                   help="vLLM scheduler concurrency. The previous 12-prompt batching gave "
                        "~186 output tok/s for a 14B model on an A100; this lets the "
                        "scheduler stay saturated.")
    p.add_argument("--max_candidates", type=int, default=None,
                   help="Cap on candidates shown to the LLM. Default None = no cap: the count "
                        "follows from the upstream candidate set (data-driven k). Set to 5 to "
                        "reproduce the original fixed top-5 behaviour.")
    p.add_argument("--min_score", type=float, default=None,
                   help="Cross-encoder score gate applied BEFORE the LLM: candidates below it "
                        "are not shown; rows with none left are recorded as NO MATCH without a "
                        "model call. Deployment: 0.9 (the same gate final_data_clean.py "
                        "applies after the LLM).")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N rows of each shard (smoke tests).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_llm_over_cross_shards(
        shards_dir=args.shards_dir,
        llm_model=args.llm_model,
        out_dir=args.out_dir,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        save_every=args.save_every,
        tensor_parallel_size=args.tensor_parallel_size,
        max_candidates=args.max_candidates,
        min_score=args.min_score,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        seed=args.seed,
        reasoning_effort=None if args.reasoning_effort == "none" else args.reasoning_effort,
        use_chat_template=not args.no_chat_template,
        prompt_file=args.prompt_file,
        threads=args.threads,
        context_json=args.context_json,
        limit=args.limit,
        candidate_layout=args.candidate_layout,
        output_format=args.output_format,
        show_scores=args.show_scores,
        hpo_json=args.hpo_json,
        hpo_multi_only=args.hpo_multi_only,
    )
