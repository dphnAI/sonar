# LMCache Examples

This folder demonstrates how to use LMCache for disaggregated prefilling, CPU offloading and KV cache sharing.

The CPU-offload example uses `LMCacheConnectorV1` in process. The
disaggregated-prefill and sharing scripts use `LMCacheMPConnector` with a
separate LMCache server. Install the connector dependencies with
`tools/install_kv_connectors.sh` from the repository root.

### MP mode and the extensible KV cache

Sonar loads `LMCacheMPConnector` from the installed LMCache package. The
maintained integration declares legacy CUDA IPC, which disables the extensible
KV cache. Do not enable `lmcache.mp.use_vmm_api` with this provider. Upstream's
file-descriptor transport requires a provider implementing `kv_cache_access`
and the LMCache file-descriptor channel, which this integration does not expose.
External LMCache module paths and the engine protocol identifier `vllm` retain
their dependency-defined spelling.

## 1. Disaggregated Prefill in Sonar v1

This example demonstrates how to run LMCache with disaggregated prefill using NIXL on a single node.

### Prerequisites

- Install [LMCache](https://github.com/LMCache/LMCache). You can simply run `pip install lmcache`.
- Install [NIXL](https://github.com/ai-dynamo/nixl).
- At least 2 GPUs
- Valid Hugging Face token (HF_TOKEN) for Llama 3.1 8B Instruct.

### Usage

Run
`cd disagg_prefill_lmcache_v1`
to get into `disagg_prefill_lmcache_v1` folder, and then run

```bash
bash disagg_example_nixl.sh
```

to run disaggregated prefill and benchmark the performance.

### Components

#### Server Scripts

- `disagg_prefill_lmcache_v1/disagg_sonar_launcher.sh` - Launches individual Sonar servers for prefill/decode, and also launches the proxy server.
- `disagg_prefill_lmcache_v1/disagg_proxy_server.py` - FastAPI proxy server that coordinates between prefiller and decoder
- `disagg_prefill_lmcache_v1/disagg_example_nixl.sh` - Main script to run the example

#### Configuration

- `disagg_prefill_lmcache_v1/configs/lmcache-prefiller-config.yaml` - Configuration for prefiller server
- `disagg_prefill_lmcache_v1/configs/lmcache-decoder-config.yaml` - Configuration for decoder server

#### Log Files

The main script generates several log files:

- `prefiller.log` - Logs from the prefill server
- `decoder.log` - Logs from the decode server
- `proxy.log` - Logs from the proxy server

## 2. CPU Offload

Run `PYTHONHASHSEED=0 python cpu_offload_lmcache.py --model Qwen/Qwen3-0.6B`.
The script configures a 5 GiB CPU cache with 256-token chunks and leaves GPU
memory utilization at Sonar's default. It disables Sonar's local prefix cache
so the printed cached-token counts show LMCache reuse.

## 3. KV Cache Sharing

Run `PYTHONHASHSEED=0 python kv_cache_sharing_lmcache_v1.py`. Both instances
use `LMCacheMPConnector` with the `kv_both` role.

## 4. Disaggregated Prefill with MP

Run `PYTHONHASHSEED=0 python disagg_prefill_lmcache.py`. The script starts an
LMCache server on port 8100 with a 5 GiB LRU tier, a producer on GPU 0, and a
consumer on GPU 1. The consumer waits for the producer's completion event
before generating. Neither script uses the removed v0 connector or a `-v`
version selector.
