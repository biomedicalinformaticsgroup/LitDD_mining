# LitDD GPU image — one image for every GPU stage of the pipeline.
#
#   BERT screen        litdd/pipeline/bert_predict_vllm.py   (vLLM pooling / classify)
#   Cross-encoder      litdd/pipeline/crossencode.py         (sentence-transformers)
#   LLM adjudication   litdd/pipeline/llm_map.py             (vLLM generate, GPT-OSS-20B)
#
# WHY `FROM vllm/vllm-openai` RATHER THAN A RAW CUDA BASE
# ------------------------------------------------------
# torch and vLLM are tightly coupled: vLLM pins the torch build, and installing them
# independently produces unresolvable pairs (the previous version of this file asked for
# torch 2.8 alongside vllm 0.10.1.1, which cannot both be satisfied — this is the substance
# of reviewer point R2-R2). Starting from the official vLLM image inherits a known-good
# torch/vLLM/transformers triple and reduces this file to the thin layer LitDD adds.
#
# WHY THIS IMAGE IS REQUIRED, NOT AN OPTIMISATION
# -----------------------------------------------
# Cluster pods have no egress (DNS returns IPv6-only records for PyPI and there is no IPv6
# route), so nothing can be pip-installed at pod start. Image *pulls* do work. Measured on
# vllm/vllm-openai:v0.23.0, the base image ships NEITHER `datasets` NOR `pyarrow` NOR
# `pandas` — all of which the screen needs, `datasets` specifically for the streaming reader
# that stops a 42 GB corpus being materialised in memory. Without this layer the corpus run
# cannot start.
#
# VLLM VERSION
# ------------
# 0.23.0 is the floor set by GPT-OSS-20B. The screen was verified on both 0.10.1.1 and
# 0.23.0, but note the pooling API was renamed between them — `task="classify"` was removed
# in favour of `runner`/`convert`. `load_vllm_engine()` negotiates both, so bumping this ARG
# does not silently break the screen; re-run revision/vllm_modernbert_check_job.yaml against
# any new version before adopting it.
ARG VLLM_VERSION=v0.23.0
FROM vllm/vllm-openai:${VLLM_VERSION}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/hf_cache

# Java is required only by the `cadmus` full-text fetcher used in the HPO-annotation stage.
RUN apt-get update && apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# The LitDD layer. Deliberately does NOT touch torch, vllm or transformers — those come from
# the base image and pinning them here is what breaks the pairing.
COPY requirements-image.txt /tmp/requirements-image.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements-image.txt

WORKDIR /opt/litdd
COPY . /opt/litdd
RUN python3 -m pip install --no-cache-dir --no-deps -e .

# Record what actually resolved, so requirements.txt can be regenerated from a build rather
# than hand-maintained, and so the run manifest can record exact versions (R2-E3).
RUN python3 -m pip freeze > /opt/litdd/containers/pip-freeze.txt \
    && python3 -c "import vllm, torch, transformers, datasets, pyarrow; \
         print('vllm', vllm.__version__); print('torch', torch.__version__); \
         print('transformers', transformers.__version__); \
         print('datasets', datasets.__version__); print('pyarrow', pyarrow.__version__)"

# The base image's entrypoint is `vllm serve`; LitDD invokes python scripts directly.
ENTRYPOINT []
CMD ["/bin/bash"]
