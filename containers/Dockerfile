# LitDD GPU image: one image for the screen (bert_predict_vllm.py) and the adjudication
# stage (llm_map.py), both under vLLM.
#
# The image starts from the official vLLM image so that torch, vLLM and transformers arrive
# as a mutually consistent set; LitDD adds only its own dependencies on top. Compute pods
# without internet access pull this image instead of installing packages at start-up.
# VLLM_VERSION 0.23.0 is the minimum for GPT-OSS-20B; load_vllm_engine() accepts the pooling
# API of both 0.10.x and 0.23.x.
ARG VLLM_VERSION=v0.23.0
FROM vllm/vllm-openai:${VLLM_VERSION}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /opt/litdd
COPY . /opt/litdd
RUN python3 -m pip install --no-cache-dir --no-deps -e .

# Record the resolved versions inside the image.
RUN python3 -m pip freeze > /opt/litdd/containers/pip-freeze.txt \
    && python3 -c "import vllm, torch, transformers, datasets, pyarrow; \
         print('vllm', vllm.__version__); print('torch', torch.__version__); \
         print('transformers', transformers.__version__); \
         print('datasets', datasets.__version__); print('pyarrow', pyarrow.__version__)"

# The base image's entrypoint is `vllm serve`; LitDD invokes python modules directly.
ENTRYPOINT []
CMD ["/bin/bash"]
