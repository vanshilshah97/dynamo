#!/usr/bin/env bash
# Start dynamo frontend with the KV router enabled.
#
# Run INSIDE the container (etcd + nats-server must already be running):
#   docker exec -d nemotron_diag bash /workspace/repo/tmp/launch_frontend.sh
#
# Log lands at /tmp/vanshils_nemotron/frontend.log and is mirrored to
# /home/scratch.vanshils_gpu_1/nemotron-turbo-nim/tmp/diag_logs/frontend.log
# (and a cleaned .clean.log) every 15s by the host-side mirror loop.
set -euo pipefail
export PYTHONHASHSEED=0
# Enable DEBUG logging on the KV router so per-request scoring formula lines surface:
#   `... with N.NN effective cached blocks: LOGIT = prefill_load_scale * ... + decode_blocks ...`
# Dynamo uses DYN_LOG (not RUST_LOG) to filter Rust tracing output.
export DYN_LOG="info,dynamo_kv_router=debug,dynamo_llm::kv_router=debug"
# Redirect both fds so docker exec -d (which discards stdout/err) doesn't lose them.
exec >/tmp/vanshils_nemotron/frontend.log 2>&1
exec python -m dynamo.frontend \
  --router-mode kv \
  --router-reset-states \
  --http-port 8000
