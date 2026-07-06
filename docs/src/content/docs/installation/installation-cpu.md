---
title: Installation with CPU
---

Aphrodite implements CPU support using multiple different backends. The most performant is
OpenVINO, but we also support inference via IPEX (Intel Extensions for PyTorch). The supported architectures are AVX2, AVX512, and PPC64LE.

The only CPU backend that supports quantization is OpenVINO, which can load FP16 Hugging Face models into INT8.

## OpenVINO Backend

### Requirements

- Linux
- Python 3.9 - 3.11
- Instruction set architecture: at least AVX2

### Dockerfile

```sh
docker build -f Dockerfile.openvino -t aphrodite-openvino .
docker run -it --rm aphrodite-openvino
```

### Building from Source

First, install Python. On Ubuntu 22.04 machines, you can run:

```sh
sudo apt-get update
sudo apt-get install python3
```

Then, install the requirements for Aphrodite:

```sh
python3 -m pip install -U pip
python3 -m pip install -r requirements/build.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Finally, install Aphrodite:

```sh
PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu" APHRODITE_TARGET_DEVICE=openvino python3 -m pip install -e .
```

### Performance tips for OpenVINO

The OpenVINO backend uses the following environment variables:

- `APHRODITE_OPENVINO_KVCACHE_SPACE` : To specify the KV cache size. e.g., 40 would mean 40GB of KV cache space. Larger numbers allows for more parallel requests. This would occupy space in RAM, so be careful.
- `APHRODITE_OPENVINO_CPU_KV_CACHE_PRECISION=u8` : Set the KV cache precision. By default, FP16/BF16 is used. This will set it to INT8.
- `APHRODITE_OPENVINO_ENABLE_QUANTIZED_WEIGHTS=ON` : To enable INT8 weights compression during model loading. By default, this is turned on. You can also export your model with different compression techniques using `optimum-cli` and pass the exported folder as the model ID to aphrodite.

To enable further performance improvements, use `--enable-chunked-prefill`. The recommend batch size for chunked prefill in OpenVINO is `--max-num-batched-tokens 256`.

### Limitations

- LoRA is not supported.
- Only decoder-only LLMs are supported. Vision and Embedding models are not.
- Tensor and Pipeline Parallelism is not supported.

## CPU Backend

We also support basic CPU inference for x86_64 platforms. The only supported data types are FP32 and BF16.

### Requirements

- Linux (or WSL on Windows)
- Compiler: gcc/g++ >= 12.3.0
- Instruct set: AVX2 or AVX512 (recommended)

### Dockerfile

```sh
docker build -f Dockerfile.cpu -t aphrodite-cpu --shm-size=4g .
docker run -it \
  --rm \
  --network=host \
  --ipc=host \
  -p 2242:2242 \
  #--cpuset-cpus=<cpu-id-list, optional> \
  #--cpuset-mems=<memory-node, optional> \
  aphrodite-cpu
```

### Building from Source

First, install the compiler to avoid potential issues.

```sh
sudo apt-get update
sudo apt-get install -y gcc-12 g++-12 libnuma-dev
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12
```

Then, install the requirements:

```sh
pip install -U pip
pip install wheel packaging ninja "setuptools>=49.4.0" numpy
uv pip install -v -r requirements/cpu.txt --torch-backend cpu
```

Finally, install Aphrodite:

```sh
APHRODITE_TARGET_DEVICE=cpu python setup.py install
```

### Intel Extension for PyTorch

You can massively boost the performance of the CPU backend by installing IPEX. Installation instructions are provided in the `Dockerfile.cpu`.

### Performance tips

- Aphrodite CPU backend uses env variable `APHRODITE_CPU_KVCACHE_SPACE` to specify the KV cache size in GBs.
- We highly recommend using TCMalloc for high performance memory allocation and better cache locality. For example, on Ubuntu 22.04 you'd run:

```sh
sudo apt-get install libtcmalloc-minimal4
sudo find / -name *libtcmalloc*
export LD_PRELOAD=/usr/lib/x86_64-linux-gpu/libtcmalloc_minimal.so.4:$LD_PRELOAD
```

- The CPU backend uses OpenMP for thread-parallel computation. If you want the best performance on CPU, it'll be very critical to isolate CPU cores for OpenMP threads with other thread pools (like web-service event-loop), to avoid CPU oversubscription.
- If using Aphrodite CPU backend on bare-metal, it's recommended to disable hyper-threading.
- If using Aphrodite CPU backend on a multi-socket machine with NUMA, make sure to set CPU cores and memory nodes, to avoid remote memory node access. `numactl` is a useful tool for this.
