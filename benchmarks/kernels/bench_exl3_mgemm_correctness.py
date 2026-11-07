# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from sonar import _custom_ops as ops


@dataclass
class CandidateResult:
    shape: int
    sms: int
    correct: bool
    max_abs: float
    ms: float | None


@dataclass
class SweepResult:
    model: str
    prefixes: list[str]
    batch: int
    outputs: int
    size_k: int
    size_n: int
    bits: int
    mcg: bool
    mul1: bool
    weighted: bool
    auto_noise_abs: float
    allowed_abs: float
    candidates: list[CandidateResult]
    best_correct: CandidateResult | None


def _load_tensor(model: str, key: str, device: str) -> torch.Tensor:
    with safe_open(f"{model}/model.safetensors", framework="pt", device="cpu") as f:
        return f.get_tensor(key).to(device=device)


def _has_tensor(model: str, key: str) -> bool:
    with safe_open(f"{model}/model.safetensors", framework="pt", device="cpu") as f:
        return key in set(f.keys())


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


def _ptr_tensor(tensors: list[torch.Tensor], device: str) -> torch.Tensor:
    return torch.tensor([tensor.data_ptr() for tensor in tensors], dtype=torch.long, device=device)


def _make_case(
    model: str,
    prefixes: list[str],
    batch: int,
    device: str,
    *,
    weighted: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    int,
    bool,
    bool,
    int,
    int,
]:
    trellises = [_load_tensor(model, f"{prefix}.trellis", device) for prefix in prefixes]
    suhs = [_load_tensor(model, f"{prefix}.suh", device) for prefix in prefixes]
    svhs = [_load_tensor(model, f"{prefix}.svh", device) for prefix in prefixes]

    first_trellis = trellises[0]
    first_suh = suhs[0]
    first_svh = svhs[0]
    assert all(tensor.shape == first_trellis.shape for tensor in trellises)
    assert all(tensor.shape == first_suh.shape for tensor in suhs)
    assert all(tensor.shape == first_svh.shape for tensor in svhs)

    mcg = _has_tensor(model, f"{prefixes[0]}.mcg")
    mul1 = _has_tensor(model, f"{prefixes[0]}.mul1")
    assert all(_has_tensor(model, f"{prefix}.mcg") == mcg for prefix in prefixes)
    assert all(_has_tensor(model, f"{prefix}.mul1") == mul1 for prefix in prefixes)

    size_k = first_trellis.shape[0] * 16
    size_n = first_trellis.shape[1] * 16
    num_outputs = len(prefixes)
    x = torch.randn((1, batch, size_k), device=device, dtype=torch.float16)
    out = torch.empty((num_outputs, batch, size_n), device=device, dtype=torch.float16)
    x_had = torch.empty((num_outputs, batch, size_k), device=device, dtype=torch.float16)
    indices = torch.arange(num_outputs, device=device, dtype=torch.long).view(1, num_outputs)
    weights = None
    if weighted:
        weights = torch.linspace(0.25, 1.0, num_outputs, device=device, dtype=torch.float16).view(1, num_outputs)
        out = torch.empty((num_outputs, batch, size_n), device=device, dtype=torch.float32)

    return (
        x,
        _ptr_tensor(trellises, device),
        out,
        _ptr_tensor(suhs, device),
        x_had,
        _ptr_tensor(svhs, device),
        indices,
        weights,
        first_trellis.shape[2] // 16,
        mcg,
        mul1,
        size_k,
        size_n,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep EXL3 MGEMM shape/SM choices with output correctness checks.")
    parser.add_argument(
        "-m",
        "--model",
        default="/home/alpindale/models/Trinity-Nano-Preview-exl3-4.0bpw",
    )
    parser.add_argument(
        "--prefix",
        action="append",
        default=None,
        help="Safetensors prefix ending before .trellis/.suh/.svh. Pass multiple times.",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--sms", type=int, nargs="*", default=None)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument(
        "--noise-multiplier",
        type=float,
        default=3.0,
        help="Allowed candidate max_abs is max(atol, auto-vs-auto noise * this value).",
    )
    parser.add_argument("--weighted", action="store_true")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Write the full correctness/timing sweep as JSON.",
    )
    args = parser.parse_args()

    device = "cuda"
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    prefixes = args.prefix or [
        "model.layers.10.mlp.experts.0.gate_proj",
        "model.layers.10.mlp.experts.0.up_proj",
    ]
    (
        x,
        ptrs_trellis,
        out,
        ptrs_suh,
        x_had,
        ptrs_svh,
        indices,
        weights,
        k,
        mcg,
        mul1,
        size_k,
        size_n,
    ) = _make_case(args.model, prefixes, args.batch, device, weighted=args.weighted)

    def run(force_shape: int = -1, force_sms: int = 0) -> torch.Tensor:
        out.zero_()
        ops.exl3_mgemm(
            x,
            ptrs_trellis,
            out,
            ptrs_suh,
            x_had,
            ptrs_svh,
            indices,
            weights,
            k,
            force_shape,
            mcg,
            mul1,
            -1,
            -1,
            force_sms,
        )
        return out

    reference = run().detach().clone()
    repeat = run().detach().clone()
    auto_noise_abs = (repeat - reference).abs().max().item()
    allowed_abs = max(args.atol, auto_noise_abs * args.noise_multiplier)
    device_sms = torch.cuda.get_device_properties(device).multi_processor_count
    sms_values = args.sms or [0, 8, 16, 24, 32, 40, 48, 56, 64, device_sms]
    sms_values = sorted({sms for sms in sms_values if sms == 0 or sms <= device_sms})

    print(f"model={args.model}")
    print(f"prefixes={prefixes}")
    print(
        f"batch={args.batch} outputs={len(prefixes)} k={size_k} n={size_n} "
        f"bits={k} mcg={mcg} mul1={mul1} weighted={args.weighted}"
    )
    print(f"auto_noise_abs={auto_noise_abs:.5f} allowed_abs={allowed_abs:.5f}")
    print("shape sms correct max_abs ms")

    candidates = [(-1, 0)]
    candidates.extend((shape, sms) for shape in range(1, 5) for sms in sms_values if sms != 0)
    results: list[CandidateResult] = []
    best: CandidateResult | None = None
    for force_shape, force_sms in candidates:

        def bench(force_shape: int = force_shape, force_sms: int = force_sms) -> None:
            run(force_shape, force_sms)

        try:
            candidate = run(force_shape, force_sms).detach().clone()
            diff = (candidate - reference).abs()
            max_abs = diff.max().item()
            correct = max_abs <= allowed_abs
            ms = _bench_ms(bench, args.warmup, args.iters) if correct else float("nan")
        except Exception as err:
            print(f"{force_shape:5d} {force_sms:3d} ERROR {type(err).__name__}: {err}")
            continue

        print(f"{force_shape:5d} {force_sms:3d} {str(correct):>7} {max_abs:8.5f} {ms:8.4f}")
        result = CandidateResult(
            shape=force_shape,
            sms=force_sms,
            correct=correct,
            max_abs=max_abs,
            ms=ms if correct else None,
        )
        results.append(result)
        if result.ms is not None and (best is None or best.ms is None or result.ms < best.ms):
            best = result

    if best is not None and best.ms is not None:
        print(f"best_correct shape={best.shape} sms={best.sms} ms={best.ms:.4f}")

    if args.json_output is not None:
        sweep = SweepResult(
            model=args.model,
            prefixes=prefixes,
            batch=args.batch,
            outputs=len(prefixes),
            size_k=size_k,
            size_n=size_n,
            bits=k,
            mcg=mcg,
            mul1=mul1,
            weighted=args.weighted,
            auto_noise_abs=auto_noise_abs,
            allowed_abs=allowed_abs,
            candidates=results,
            best_correct=best,
        )
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(asdict(sweep), indent=2) + "\n")


if __name__ == "__main__":
    main()
