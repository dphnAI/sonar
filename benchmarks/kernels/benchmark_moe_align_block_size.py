# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import itertools

import torch

import sonar._custom_ops as ops
from sonar.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from sonar.platforms import current_platform
from sonar.triton_utils import triton


def get_topk_ids(num_tokens: int, num_experts: int, topk: int) -> torch.Tensor:
    return torch.stack(
        [torch.randperm(num_experts, dtype=torch.int32, device="cuda")[:topk] for _ in range(num_tokens)]
    )


# test configurations
num_tokens_range = [1, 16, 256, 4096]
num_experts_range = [16, 64, 224, 256, 280, 512]
topk_range = [1, 2, 8]
configs = list(itertools.product(num_tokens_range, num_experts_range, topk_range))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens", "num_experts", "topk"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["sonar"],
        line_names=["Sonar"],
        plot_name="moe-align-block-size-performance",
        args={},
    )
)
def benchmark(num_tokens, num_experts, topk, provider):
    """Benchmark function for Triton."""
    block_size = 256
    topk_ids = get_topk_ids(num_tokens, num_experts, topk)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "sonar":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: moe_align_block_size(topk_ids, block_size, num_experts),
            quantiles=quantiles,
        )

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


batched_configs = list(
    itertools.product(
        [8, 32, 64, 256],
        [16, 256, 512, 2048],
        [8, 32, 48, 64],
        ["uniform", "skewed", "mostly_empty"],
    )
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=[
            "num_batches",
            "max_tokens_per_batch",
            "block_size",
            "distribution",
        ],
        x_vals=batched_configs,
        line_arg="provider",
        line_vals=["sonar"],
        line_names=["Sonar"],
        plot_name="batched-moe-align-block-size-performance",
        args={},
    )
)
def benchmark_batched(
    num_batches,
    max_tokens_per_batch,
    block_size,
    distribution,
    provider,
):
    current_platform.seed_everything(0)
    if distribution == "uniform":
        expert_num_tokens = torch.full(
            (num_batches,),
            max_tokens_per_batch,
            dtype=torch.int32,
            device="cuda",
        )
    elif distribution == "skewed":
        expert_num_tokens = torch.ones((num_batches,), dtype=torch.int32, device="cuda")
        expert_num_tokens[0] = max_tokens_per_batch
    else:
        expert_num_tokens = torch.zeros((num_batches,), dtype=torch.int32, device="cuda")
        expert_num_tokens[::8] = max_tokens_per_batch

    max_num_tokens_padded = num_batches * ((max_tokens_per_batch + block_size - 1) // block_size) * block_size
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device="cuda")
    expert_ids = torch.empty((max_num_tokens_padded // block_size,), dtype=torch.int32, device="cuda")
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device="cuda")

    quantiles = [0.5, 0.2, 0.8]
    if provider == "sonar":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: ops.batched_moe_align_block_size(
                max_tokens_per_batch,
                block_size,
                expert_num_tokens,
                sorted_ids,
                expert_ids,
                num_tokens_post_pad,
            ),
            quantiles=quantiles,
        )

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batched", action="store_true")
    parser.add_argument(
        "--num_experts",
        type=int,
        default=64,
        choices=[8, 16, 32, 64, 128, 256],
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=8,
        choices=[2, 4, 8],
        help="Top-k value for correctness check.",
    )
    args = parser.parse_args()

    selected_benchmark = benchmark_batched if args.batched else benchmark
    selected_benchmark.run(print_data=True, show_plots=True)
