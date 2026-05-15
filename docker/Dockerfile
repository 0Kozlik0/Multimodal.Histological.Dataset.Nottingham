# syntax=docker/dockerfile:1
# MIL baseline: PyTorch + OpenSlide + Python deps for WSI patching and feature extraction.
#
# Official PyTorch images: https://hub.docker.com/r/pytorch/pytorch/tags
# Override when building if you need another CUDA/PyTorch combo, e.g.:
#   docker build --build-arg PYTORCH_IMAGE=pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime -f docker/Dockerfile .
ARG PYTORCH_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

USER root

# OpenSlide (.svs / .mrxs) headers + HTTPS for pip/HF
RUN apt-get update --fix-missing \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        apt-transport-https \
        ca-certificates \
        libopenslide-dev \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/Custom.Dataset.Nottingham

# Install Python dependencies and the local package (matches pyproject + src layout)
COPY requirements.txt pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e .
