#!/usr/bin/env bash
set -euo pipefail
export PYTHONHASHSEED=0
export VLLM_LOGGING_LEVEL=INFO
cd /tmp/vanshils_nemotron
exec >/tmp/vanshils_nemotron/worker.log 2>&1
exec python -m dynamo.vllm \
  --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 \
  --tensor-parallel-size 8 \
  --block-size 64 \
  --trust-remote-code \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}'
