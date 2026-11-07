FROM python:3.12-slim

# git: manifests record the commit; build-essential: some wheels compile from source
RUN apt-get update \
 && apt-get install -y --no-install-recommends git build-essential \
 && rm -rf /var/lib/apt/lists/* \
 && git config --system --add safe.directory /workspace

# uv installs Python 3.11 and creates the venvs quickly.
# Pinned tag: bump it deliberately, never use "latest".
COPY --from=ghcr.io/astral-sh/uv:0.8.0 /uv /usr/local/bin/uv

COPY mlops/requirements.txt /tmp/mlops-req.txt
COPY SageServe-original/requirements.txt /tmp/sim-req.txt

# Environment 1: MLOps pipeline (Python 3.12)
RUN uv venv --python 3.12 /opt/venvs/mlops \
 && uv pip install --python /opt/venvs/mlops/bin/python -r /tmp/mlops-req.txt

# Environment 2: SageServe simulator (Python 3.11)
RUN uv venv --python 3.11 /opt/venvs/sim \
 && uv pip install --python /opt/venvs/sim/bin/python -r /tmp/sim-req.txt

# Environment 3: Streamlit web UI
RUN uv venv --python 3.12 /opt/venvs/app \
 && uv pip install --python /opt/venvs/app/bin/python streamlit pandas pyyaml

# Path variables used by the helper commands and venv
ENV REPO=/workspace \
    MLOPS_DIR=/workspace/mlops \
    SIM_DIR=/workspace/SageServe-original \
    MLOPS_DATA_ROOT=/workspace/data \
    MODEL_REGISTRY_DIR=/workspace/data/model_registry \
    MLOPS_PYTHON=/opt/venvs/mlops/bin/python \
    SIM_PYTHON=/opt/venvs/sim/bin/python \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/workspace/docker/bin:/opt/venvs/mlops/bin:${PATH}"

# Load the shell shortcuts in every interactive shell.
# The repo itself is mounted at runtime, so editing docker/ needs no rebuild.
RUN echo '[ -f /workspace/docker/sageserve.sh ] && . /workspace/docker/sageserve.sh' >> /etc/bash.bashrc

WORKDIR /workspace
EXPOSE 8501
CMD ["bash"]