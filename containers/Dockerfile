# Docker image for the LitDD GPU stack (portable alternative to containers/litdd.def).
# Same contents as the Apptainer def: CUDA 12.9 runtime + Java + pinned torch/vLLM stack.
#
#   docker build -f containers/Dockerfile -t litdd:latest .
#   docker run --gpus all -it litdd:latest python annotate_pubmed/bert_predict_vllm.py --help
FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 HF_HOME=/tmp/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv git \
        openjdk-17-jre-headless ca-certificates wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/litdd
COPY requirements.txt /opt/litdd/requirements.txt

# torch against the CUDA 12.9 wheels, then the rest of the pinned stack
RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir torch==2.8.* --index-url https://download.pytorch.org/whl/cu129 \
    && python3 -m pip install --no-cache-dir -r /opt/litdd/requirements.txt

COPY . /opt/litdd
