# Qwen3.5-122B Recipes

Recipes for [Qwen/Qwen3.5-122B-A10B-FP8](https://huggingface.co/Qwen/Qwen3.5-122B-A10B-FP8),
the FP8 checkpoint of [Qwen/Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B)
(122B total / 10B active hybrid MoE — Gated DeltaNet linear attention + MoE with
full attention every 4th layer). The FP8 weights fit a single 143 GB H200 at the
full 262,144-token context, which is what this recipe exploits.

## Configurations

Dynamo + vLLM deployment profile for the agentic workload on **H200**.

|                          | H200 aggregated agentic                          |
| ------------------------ | ------------------------------------------------ |
| **GPU**                  | 1x H200 per worker; scale via `replicas`         |
| **Mode**                 | Aggregated                                       |
| **Framework**            | vLLM                                             |
| **Precision**            | FP8 weights + BF16 KV                            |
| **Parallelism**          | TP1                                              |
| **KV cache manager**     | Hybrid (DeltaNet SSM + attention)                |
| **Routing**              | KV-aware (`DYN_ROUTER_MODE=kv`)                  |
| **Speculative decoding** | None — see Notes for the optional MTP variant    |
| **Context length**       | 262,144 (model default)                          |
| **KV transfer**          | N/A                                              |

### Why TP1 + replicas

Every multi-GPU engine layout measured (TP2, TP4, TP8, DP+EP) delivered less
output throughput **per GPU** than independent TP1 replicas at the agentic SLA.
The winning layout is one engine per GPU, scaled horizontally behind the
KV-aware router. KV routing is load-bearing: the replicas are independent
engines and agentic requests share ~57k-token prefixes, so the router must
land each request on the replica that already holds its prefix.

The DGD ships with `replicas: 2` for the agg worker (a minimal KV-router
validation deploy); a full 8x H200 node runs `replicas: 8`.

## Supported features

- Modalities: Text
- Reasoning
- Tool calling

## Prerequisites

1. **Dynamo Platform installed** on the target cluster with DGD CRDs served —
   see [Kubernetes Deployment Guide](../../docs/kubernetes/README.md).
2. **NGC/nvcr image pull access** — the deploy manifest pulls from
   `nvcr.io/nvstaging/ai-dynamo`; if the cluster does not inject a default pull
   secret, create one and attach it to the namespace's default service account.
3. **Hugging Face token** with access to `Qwen/Qwen3.5-122B-A10B-FP8`, stored
   as `hf-token-secret` — used by both the model-download Job and the serving
   workers.
4. **`model-cache` PVC** (ReadWriteMany) populated with the model, or permission
   to create and populate it via the manifests in `model-cache/`.

## Quick Start

### 1. Create namespace and secret

```bash
export NAMESPACE=your-namespace
kubectl create namespace ${NAMESPACE}
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="your-token" \
  -n ${NAMESPACE}
```

> [!NOTE]
> If the target namespace lacks nvcr/NGC pull access, create a pull secret and
> attach it to the default service account:
>
> ```bash
> kubectl create secret docker-registry nvcr-secret \
>   --docker-server=nvcr.io --docker-username='$oauthtoken' \
>   --docker-password="<your-NGC-API-key>" -n ${NAMESPACE}
> kubectl patch serviceaccount default -n ${NAMESPACE} \
>   -p '{"imagePullSecrets":[{"name":"nvcr-secret"}]}'
> ```

### 2. Create storage

> [!NOTE]
> Edit `model-cache/model-cache.yaml` and set `storageClassName` to a
> ReadWriteMany storage class available on the target cluster.

```bash
kubectl apply -f model-cache/model-cache.yaml -n ${NAMESPACE}
```

### 3. Download the model

```bash
kubectl apply -f model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=7200s
```

### 4. Deploy the DGD

```bash
kubectl apply -f vllm/agg-h200/deploy.yaml -n ${NAMESPACE}
```

To use a full 8x H200 node, edit `spec.components[agg].replicas` to `8` before
applying.

### 5. Benchmark

See [perf/README.md](perf/README.md) for the full benchmark workflow — trace
staging on the PVC, running the AIPerf trace-replay Job, running a concurrency
sweep, and fetching artifacts.

## Optimization targets

| Workload | Median ISL | Median OSL | KV cache hit rate | User output tok/s |
| -------- | ---------- | ---------- | ----------------- | ----------------- |
| Agentic  | 64k        | 400        | 90%               | 50                |

## Performance results

Measured on 8x H200 against the real 15% agentic mooncake trace (closed-loop
concurrency; SLA = ≥ 50 output tok/s/user). Headline metric is system output
tok/s per GPU at the best SLA-passing concurrency.

| Recipe                | GPUs | tok/s/GPU @ SLA |
| --------------------- | ---- | --------------- |
| Aggregated TP1 (x8)   | 8    | 264             |

TP2, TP4, TP8, and DP+EP layouts of the same engine were all measured and all
delivered lower tok/s/GPU at the SLA — TP1 replicas + KV-aware routing is the
recommended configuration.

## Notes

- **`--max-num-seqs` must stay ≤ 228.** The Mamba/DeltaNet SSM cache is
  block-allocated at TP1 and the vLLM default (`1024`) crashes at startup.
  The recipe ships `128`, the measured sweet spot for this workload.
- **`--kv-cache-dtype auto` (BF16 KV) is intentional.** The FP8 checkpoint
  ships no KV scales; BF16 KV also measured best for this recipe.
- **Optional MTP speculative decoding (unbenchmarked).** MTP works with
  `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'` but
  requires raising `--gpu-memory-utilization` to `0.95` at the full 262k
  context. It is not part of the measured recipe.
- **Runtime version.** The manifest pins
  `nvcr.io/nvstaging/ai-dynamo/vllm-runtime:1.3.0-rc.11` — the exact image the
  performance study was measured on (vLLM 0.23.0).
