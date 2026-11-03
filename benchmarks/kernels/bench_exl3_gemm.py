# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
from collections.abc import Callable

import torch
from safetensors import safe_open

from aphrodite import _custom_ops as ops


def _load_tensor(model: str, key: str, device: str) -> torch.Tensor:
    with safe_open(f"{model}/model.safetensors", framework="pt", device="cpu") as f:
        return f.get_tensor(key).to(device=device)


def _bench_ms(fn: Callable[[], None], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _make_case(model: str, prefix: str, batch: int, device: str):
    trellis = _load_tensor(model, f"{prefix}.trellis", device)
    suh = _load_tensor(model, f"{prefix}.suh", device)
    svh = _load_tensor(model, f"{prefix}.svh", device)

    with safe_open(f"{model}/model.safetensors", framework="pt", device="cpu") as f:
        keys = set(f.keys())
    mcg = f"{prefix}.mcg" in keys
    mul1 = f"{prefix}.mul1" in keys

    k = trellis.shape[0] * 16
    n = trellis.shape[1] * 16
    x = torch.randn((batch, k), device=device, dtype=torch.float16)
    out = torch.empty((batch, n), device=device, dtype=torch.float16)
    x_had = torch.empty_like(x)
    return x, trellis, out, suh, x_had, svh, mcg, mul1, k, n


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep EXL3 GEMM kernel shape/SM settings.")
    parser.add_argument(
        "-m",
        "--model",
        default="/home/alpindale/models/Trinity-Nano-Preview-exl3-4.0bpw",
    )
    parser.add_argument(
        "--prefix",
        default="model.layers.10.mlp.experts.0.down_proj",
        help="Safetensors prefix ending before .trellis/.suh/.svh",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--sms", type=int, nargs="*", default=None)
    args = parser.parse_args()

    device = "cuda"
    torch.set_grad_enabled(False)
    device_sms = torch.cuda.get_device_properties(device).multi_processor_count
    sms_values = args.sms or [0, 16, 24, 32, 40, 48, 56, 64, device_sms]
    sms_values = sorted({sms for sms in sms_values if sms == 0 or sms <= device_sms})
    x, trellis, out, suh, x_had, svh, mcg, mul1, k, n = _make_case(args.model, args.prefix, args.batch, device)

    print(f"model={args.model}")
    print(f"prefix={args.prefix}")
    print(f"batch={args.batch} k={k} n={n} bits={trellis.shape[2] // 16} mcg={mcg} mul1={mul1}")
    print("shape sms ms tflops")

    best: tuple[float, int, int] | None = None
    candidates = [(-1, 0)]
    candidates.extend((shape, sms) for shape in range(1, 5) for sms in sms_values if sms != 0)
    for force_shape, force_sms in candidates:

        def run(force_shape: int = force_shape, force_sms: int = force_sms) -> None:
            ops.exl3_gemm(
                x,
                trellis,
                out,
                suh,
                x_had,
                svh,
                force_shape,
                mcg,
                mul1,
                force_sms,
            )

        try:
            ms = _bench_ms(run, args.warmup, args.iters)
        except Exception as err:
            print(f"{force_shape:5d} {force_sms:3d} ERROR {type(err).__name__}: {err}")
            continue

        tflops = (2 * args.batch * k * n) * 1e-12 / (ms * 1e-3)
        print(f"{force_shape:5d} {force_sms:3d} {ms:8.4f} {tflops:8.3f}")
        if best is None or ms < best[0]:
            best = (ms, force_shape, force_sms)

    if best is not None:
        print(f"best shape={best[1]} sms={best[2]} ms={best[0]:.4f}")


if __name__ == "__main__":
    main()
