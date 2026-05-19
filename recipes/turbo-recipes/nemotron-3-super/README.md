<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Nemotron-3-Super Turbo NIM

Layers three vLLM source patches onto a locally-built `dynamo:latest-vllm-runtime`
image so the Dynamo KV-aware router actually receives prefix signal when
serving [nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8)
(a hybrid Mamba+Attention model).

Patch-level detail and version-skew handling:
`/home/scratch.vanshils_gpu_1/nemotron-turbo-nim/tmp/patches_for_distribution/`
(`README.md` + `COMPATIBILITY.md`).

## Prerequisites

- A node with Docker and an NVIDIA runtime.
- Local image `dynamo:latest-vllm-runtime` already built from this repo:
  ```bash
  container/render.py --framework vllm --target runtime --output-short-filename
  docker build -t dynamo:latest-vllm-runtime -f container/rendered.Dockerfile .
  ```
- `NGC_API_KEY` and `HF_TOKEN` set for HF gated download of
  `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8`.

## Build

From the dynamo repo root:

```bash
docker build \
  -t nemotron-3-super-turbo:dev \
  -f recipes/turbo-recipes/nemotron-3-super/Dockerfile \
  recipes/turbo-recipes/nemotron-3-super
```

The build-time check
`assert hasattr(EngineCoreProc, "get_kv_cache_group_metadata")` is the canary
that all three patches landed cleanly. Build fails if it doesn't.

## Run

Start the container:

```bash
docker run -it --runtime nvidia --gpus all --network host --ipc=host \
  -v /home/scratch.vanshils_gpu_1/:/my_scratch_space_1/ \
  -e NGC_API_KEY -e HF_TOKEN \
  -e HF_HOME=/my_scratch_space_1/nemotron-turbo-nim/tmp/hf_cache \
  --entrypoint /bin/bash \
  nemotron-3-super-turbo:dev
```

Inside the container:

```bash
mkdir -p /tmp/vanshils_nemotron
etcd --data-dir /tmp/etcd-data --listen-client-urls http://0.0.0.0:2379 \
     --advertise-client-urls http://0.0.0.0:2379 &
nats-server -js &

bash /workspace/recipes/turbo-recipes/nemotron-3-super/launch_worker.sh &
bash /workspace/recipes/turbo-recipes/nemotron-3-super/launch_frontend.sh &
```

Smoke test:

```bash
curl http://localhost:8000/v1/models
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
       "messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

## What the launch scripts do

- **`launch_worker.sh`** — `python -m dynamo.vllm --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 --tensor-parallel-size 8 --block-size 64 --trust-remote-code --kv-events-config '{"publisher":"zmq",...,"endpoint":"tcp://*:20080"}'`. `--block-size 64` is the `hash_block_size` Patch 02 preserves through the hybrid `attn_block_size` inflation — without it vLLM bumps to 2176 and no `BlockStored` event fires for prompts under that length.
- **`launch_frontend.sh`** — `python -m dynamo.frontend --router-mode kv --router-reset-states --http-port 8000` with `DYN_LOG=info,dynamo_kv_router=debug,dynamo_llm::kv_router=debug` so per-request KV-router scoring lines surface in `frontend.log`.

## File Layout

```text
recipes/turbo-recipes/nemotron-3-super/
  README.md
  Dockerfile
  launch_worker.sh
  launch_frontend.sh
  patches/
    01_vllm_pr40984.patch
    02_patch_c_v21_sub_block_emit.patch
    03_vllm_metadata_hash_block_size.patch
```
