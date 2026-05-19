# syntax=docker/dockerfile:1.10.0
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Nemotron-3-Super-120B-A12B-FP8 turbo NIM recipe. Layers three vLLM source
# patches (KV-event metadata, sub-block emission for hybrid Mamba/Attn,
# hash_block_size exposure through get_kv_cache_group_metadata) onto a
# locally-built dynamo vllm-runtime image so the Dynamo KV-aware router can
# index per-prefix worker affinity on a hybrid Mamba/Attention model.
#
# Build (from the dynamo repo root, on a node with docker + the local
# dynamo:latest-vllm-local image already built):
#   docker build \
#     -t nemotron-3-super-turbo:dev \
#     -f recipes/turbo-recipes/nemotron-3-super/Dockerfile \
#     recipes/turbo-recipes/nemotron-3-super

ARG BASE_IMAGE="dynamo:latest-vllm-local"

FROM ${BASE_IMAGE}
USER root

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends patch \
 && rm -rf /var/lib/apt/lists/*

COPY --chown=root:root patches/ /opt/dynamo/turbo-patches/

RUN set -eux; \
    VLLM_PARENT="$(python3 -c 'import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))')"; \
    cd "${VLLM_PARENT}"; \
    patch -p1 < /opt/dynamo/turbo-patches/01_vllm_pr40984.patch; \
    patch -p1 < /opt/dynamo/turbo-patches/02_patch_c_v21_sub_block_emit.patch; \
    patch -p1 < /opt/dynamo/turbo-patches/03_vllm_metadata_hash_block_size.patch; \
    python3 -c 'import vllm.v1.engine.core as c; assert hasattr(c.EngineCoreProc, "get_kv_cache_group_metadata"), "patch 01/03 did not land"'

COPY --chmod=755 --chown=dynamo:0 launch_worker.sh launch_frontend.sh /workspace/recipes/turbo-recipes/nemotron-3-super/

USER dynamo
