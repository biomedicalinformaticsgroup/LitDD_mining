import argparse
import gc
import glob
import os
import re

import numpy as np
import pandas as pd

# NOTE: torch and vllm are imported lazily inside run_llm_over_cross_shards so the
# deterministic helpers (prompt building, answer parsing, sharding) can be imported and
# unit-tested without the GPU stack installed.


def build_llm_prompt(tiab, candidate_lines):
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
    return (
            f"""System/Developer Instruction:
        You are an expert in genetic disease, and mapping a title+abstract (TIAB) to one or more specific G2P LGMDE threads. You will receive:
        - A TIAB
        - {n} candidate LGMDE {plural}, numbered 1-{n} (each includes its G2P ID, gene(s), allelic requirement, inheritance, mechanism, evidence, disease name)

        Goal:
        Return the G2P ID(s) from the provided candidates that best match the TIAB, or NO MATCH if none apply.

        Critical constraints:
        - Only choose from the {n} candidate(s) listed below. Do not invent any other ID.
        - Prefer selecting at least one candidate over NO MATCH unless the TIAB is clearly non-human only, describes somatic disease only, or references no overlapping gene(s) with the candidates.
        - Output exactly one line in the specified schema and nothing else.

        Decision rubric (apply in this order):
        1) Extract from the TIAB:
        - Human evidence: Does it describe human patients (case(s), cohort)? 
            - If only non-human models (mouse, zebrafish, cell lines) with no human patients, even if non-human model relates to a human disease, this is NO MATCH. 
        - Type of disease: Germline disease only. If only somatic cancer described in the TIAB, this is NO MATCH, unless there is evidence this is part of a developmental syndrome. For example, genetic variants in hepatocellular cancer are likely to be somatic, even in human subjects. Alteratively mention of Juvenile Myelomonocytic Leukemia with Noonan syndrome is part of a wider syndromic developmental disorder.
        - Type of study: If polymorphism or GWAS or genome-wide association study explictly mentioned, this is NO MATCH
        - Gene(s): exact gene symbols and aliases. Ignore vague gene families unless the exact gene symbol is present.
        - Inheritance/allelic clues: autosomal recessive/dominant, X-linked, biallelic, homozygous, compound heterozygous, de novo, heterozygous, multiplex families, consanguinity.
        - Disease name(s) and synonyms.
        - Key phenotypes (organ systems, hallmark features).

        2) Candidate screening (must pass to be considered):
        - Gene: TIAB must mention at least one gene that exactly matches a candidate gene (allow common aliases). If no gene overlap with any candidate, return NO MATCH.
        - Human: TIAB must include human subjects or clear human diagnostic statements. If absent and only non-human, return NO MATCH.
        - Disease type: TIAB must not describe somatic variation e.g. in cancer, or polymorphisms in GWAS for common diseases. 
        - Negation: TIAB must not describe a negative association e.g. Variants in gene X do not cause disease Y. 

        3) Evidence scoring per candidate (use to rank):
        - Gene match: required.
        - Allelic requirement:
            - If TIAB explicitly states zygosity/inheritance, it must be compatible with the candidate.
            - If TIAB does not state zygosity/inheritance, do NOT reject the candidate; instead rely on disease name, inheritance words (if present), and phenotype overlap to disambiguate.
            - If there are two candidate matches for a TIAB without zygosity/inheritance, choose the most common match. For example, if Marfan syndrome is mentioned it is much more likely to be the monoallelic form (common) than the biallelic form (very rare).
        - Disease name/synonym: strong positive evidence if the TIAB mentions the same disease name or clear synonym (including eponyms).
        - Phenotype: positive if hallmark/system-level features align (partial matches acceptable).
            - If the phenotype does not match but the gene and allelic requirement clearly match, consider returning the matching candidate anyway, as this may indicate differences in disease-gene curation rather than the underlying molecular basis of disease.
            - For example, PDHA1 may be PDHA1-related intellectual disability monoallelic_X_hemizygous or PDHA1-related pyruvate dehydrogenase E1-alpha deficiency monoallelic_X_heterozygous.
            - In this case, if the tiab mentions PDHA1 variants in boys, it is more important that the gene and allelic requirement match than there is an exact match to the phenotype/disease name.
        - Title emphasis: features in the title or opening sentence weigh more.

        4) Selection:
        - If exactly one candidate has a gene match and either:
            a) explicit allelic requirement match, or
            b) disease name/synonym match, or
            c) ≥2 hallmark phenotypic features match,
            return this candidate.
        - If multiple candidates share the same gene:
            - Use explicit allelic statements (if present) to disambiguate; else use disease name/synonyms; else use phenotype; else use inheritance words (AR/AD/X-linked); else prefer what the title emphasizes.
        - If the TIAB clearly describes multiple matching diseases/genes among the candidates, return all matching IDs (semicolon-separated).
        - Only return NO MATCH if:
            - No gene overlap with any candidate, or
            - The abstract is non-human only (no human patients), or
            - The evidence is clearly incompatible (e.g., explicit dominant in TIAB vs strict biallelic candidate) for all candidates.

        Output schema (strict):
        - Return exactly one line:
        ANSWER: G2PID
        or ANSWER: G2PID;G2PID
        or ANSWER: NO MATCH

    TIAB:
    {tiab}

    Candidate LGMDE Threads:
    """
            + "\n".join(f"{i+1}) {c}" for i, c in enumerate(candidate_lines))
            + "\nReturn exactly one line in the schema above."
    )


def extract_last_answer(text):
    matches = re.findall(r'ANSWER:\s*(.*)', text or "")
    return matches[-1].strip() if matches else None


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


def run_llm_over_cross_shards(
    shards_dir,
    llm_model,
    out_dir=None,
    batch_size=32,
    temperature=0.0,
    top_p=1.0,
    max_tokens=2048,
    shard_index=None,
    num_shards=None,
    save_every=1000,
    tensor_parallel_size=None,
    max_candidates=None,
    max_num_seqs=512,
):
    """
    - Reads *.parquet from shards_dir
    - Builds prompts from 'tiab' and 'top5_cross'
    - Runs vLLM in batches
    - Incrementally writes output parquet per shard every `save_every` rows,
      overwriting the same file each time:
        columns added: 'top_5_cross_lgmde', 'llm_prompt', 'generated_text', 'llm_dis_map'
    - Shard-aware: if shard_index/num_shards are provided, each worker processes
      its subset of shard files (index % num_shards == shard_index).
    """
    import torch
    from vllm import LLM, SamplingParams

    os.makedirs(out_dir or shards_dir, exist_ok=True)
    out_dir = out_dir or shards_dir

    # Initialize LLM once for all shards
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    # The prompt is a ~5,000-character fixed rubric plus a short per-record suffix, so the
    # shared prefix dominates. Without prefix caching that rubric is re-prefilled once per
    # record -- ~10^9 redundant prefill tokens over a full corpus.
    llm_kwargs = {"enable_prefix_caching": True, "max_num_seqs": max_num_seqs}
    if tensor_parallel_size is not None:
        llm_kwargs["tensor_parallel_size"] = int(tensor_parallel_size)
    llm = LLM(model=llm_model, **llm_kwargs)

    shard_paths = sorted(glob.glob(os.path.join(shards_dir, "*.parquet")))

    print(f"Found {len(shard_paths)} parquet shard(s) for this worker.")

    for shard_path in shard_paths:
        print(f"Processing shard: {os.path.basename(shard_path)}")
        df = pd.read_parquet(shard_path)
        if num_shards and num_shards > 1:
            keep = row_slice_for_worker(len(df), shard_index, num_shards)
            df = df.iloc[keep].reset_index(drop=True)
            print(f"[shard {shard_index}/{num_shards}] rows for this worker: {len(df)}")
        if df.empty:
            print("  (no rows for this worker in this file)")
            continue

        # Normalize and create list of LGMDE strings (up to 5) for the prompt
        def to_labels(x, max_candidates=max_candidates):
            """Normalise a top-k cell into a list of candidate label strings.

            max_candidates=None keeps every candidate, which is what the data-driven
            configuration wants: the number of candidates follows from the genes actually
            mentioned in the TIAB, so it is not fixed at 5.
            """
            # Normalize None/NaN
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return []

            # Helper to normalize one item to a label string
            def item_to_label(item):
                # dict-like
                if isinstance(item, dict):
                    return str(item.get("label", "")).strip() or None

                # tuple/list like (label, score)
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    return str(item[0]).strip() or None

                # PyArrow scalars/structs -> convert to Python
                try:
                    import pyarrow as pa
                    if isinstance(item, pa.Scalar):
                        item = item.as_py()
                        if isinstance(item, dict):
                            return str(item.get("label", "")).strip() or None
                        if isinstance(item, (list, tuple)) and len(item) >= 1:
                            return str(item[0]).strip() or None
                except Exception:
                    pass

                # Fallback: plain string
                if isinstance(item, str):
                    s = item.strip()
                    return s or None

                return None

            # If it’s already a list/tuple/np.ndarray, iterate
            if isinstance(x, (list, tuple, np.ndarray)):
                labels = []
                for it in (x.tolist() if isinstance(x, np.ndarray) else x):
                    lab = item_to_label(it)
                    if lab:
                        labels.append(lab)
                return labels if max_candidates is None else labels[:max_candidates]

            # If it’s a string, try JSON then literal_eval
            if isinstance(x, str):
                import ast
                import json
                obj = None
                try:
                    obj = json.loads(x)
                except Exception:
                    try:
                        obj = ast.literal_eval(x)
                    except Exception:
                        return []
                return to_labels(obj)

            # PyArrow List/Struct scalars at the top level
            try:
                import pyarrow as pa
                if isinstance(x, pa.Scalar):
                    return to_labels(x.as_py())
            except Exception:
                pass

            return []

    
        df["topk_cross_lgmde"] = df["top5_cross"].apply(to_labels)
        # Legacy alias: existing analyses (cascade_funnel, check_llm_data) read this name.
        df["top_5_cross_lgmde"] = df["topk_cross_lgmde"]


        # Build prompts
        df["llm_prompt"] = df.apply(
            lambda row: build_llm_prompt(
                tiab=row.get("tiab", ""),
                candidate_lines=row.get("topk_cross_lgmde", []),
            ),
            axis=1,
        )

        # temp just to check working
        print("Sample candidates:", df["topk_cross_lgmde"].iloc[0] if len(df) else [])
        print("Prompt preview:\n", df["llm_prompt"].iloc[0][:800])

        N = len(df)
        print(f"Total rows in shard: {N}")

        base = os.path.splitext(os.path.basename(shard_path))[0]
        suffix = "" if shard_index is None else f"_w{shard_index}"
        out_parquet = os.path.join(out_dir, f"{base}{suffix}__llm.parquet")

        # Resume: on a preemptible cluster a multi-hour shard will be interrupted, and the
        # previous implementation restarted every shard from row 0.
        generated_texts = [""] * N
        if os.path.exists(out_parquet):
            try:
                prev = pd.read_parquet(out_parquet, columns=["generated_text"])
                if len(prev) == N:
                    generated_texts = ["" if pd.isna(t) else str(t)
                                       for t in prev["generated_text"].tolist()]
                    done = sum(1 for t in generated_texts if t)
                    print(f"[RESUME] {done}/{N} rows already generated in {out_parquet}")
                else:
                    print(f"[RESUME] row-count mismatch ({len(prev)} != {N}); starting fresh")
            except Exception as e:  # noqa: BLE001 - a corrupt checkpoint must not block the run
                print(f"[RESUME] could not read {out_parquet} ({e}); starting fresh")

        def save_progress():
            df["generated_text"] = generated_texts
            df["llm_dis_map"] = [extract_last_answer(t) for t in generated_texts]
            df.to_parquet(out_parquet, index=False)
            print(f"[PROGRESS] Saved current progress to {out_parquet}", flush=True)

        todo = [i for i in range(N) if not generated_texts[i]]
        if not todo:
            print("[SKIP] shard already complete")
            continue

        # One generate() call per checkpoint window, not per `batch_size` rows.
        #
        # The previous code called llm.generate() on 12 prompts at a time, which defeats
        # vLLM's continuous batching: each call spins the scheduler up from zero, runs at
        # concurrency 12 and drains, so the KV cache never fills. The deployed run measured
        # ~186 output tokens/s for a dense 14B model on an 80GB A100 -- roughly an order of
        # magnitude below what that hardware does. Handing vLLM the whole window lets it
        # schedule continuously; `save_every` now controls only checkpoint granularity.
        window = save_every if save_every and save_every > 0 else N
        for w_start in range(0, len(todo), window):
            idx = todo[w_start:w_start + window]
            prompts = [df["llm_prompt"].iloc[i] for i in idx]
            print(f"  generating {len(prompts)} prompt(s) "
                  f"[{w_start + len(idx)}/{len(todo)} of this shard's outstanding rows]",
                  flush=True)
            outputs = llm.generate(prompts, sampling_params)
            for i, out in zip(idx, outputs):
                generated_texts[i] = out.outputs[0].text
            save_progress()

        print(f"[DONE] Shard completed: {os.path.basename(shard_path)}")

        # Optional: free some memory between shards (vLLM keeps its KV cache though)
        torch.cuda.empty_cache()
        gc.collect()

    print("All shards processed.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--shards_dir", required=True, type=str)
    p.add_argument("--llm_model", required=True, type=str)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Mapping is treated as deterministic; default 0.0.")
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max_tokens", type=int, default=2048)
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
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_llm_over_cross_shards(
        shards_dir=args.shards_dir,
        llm_model=args.llm_model,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        save_every=args.save_every,
        tensor_parallel_size=args.tensor_parallel_size,
        max_candidates=args.max_candidates,
        max_num_seqs=args.max_num_seqs,
    )
