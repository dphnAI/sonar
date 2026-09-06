# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Aphrodite project
"""First-call correctness of the EXL3 MoE single-token and small-batch paths.

``exl3_mgemm`` reads ``A`` and writes the Hadamard-transformed copy to
``A_had``. The cooperative-kernel autotuner re-launches the kernel on the same
arguments while it times candidates, so if ``A_had`` aliases ``A`` every launch
after the first transforms the previous launch's output: the first call for a
shape returns garbage, and later calls (served from the autotune cache) are
correct. The down projection used to pass ``exl3_small_interm_a`` for both.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from aphrodite import _custom_ops as ops
from aphrodite.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip(reason="EXL3 requires CUDA", allow_module_level=True)

DEVICE = "cuda"
K_BITS = 3


def _trellis(size_k: int, size_n: int) -> torch.Tensor:
    return torch.randint(-(2**15), 2**15, (size_k // 16, size_n // 16, 16 * K_BITS), dtype=torch.int16, device=DEVICE)


def _signs(size: int) -> torch.Tensor:
    return torch.where(torch.rand(size, device=DEVICE) < 0.5, -1.0, 1.0).to(torch.float16)


def _sylvester_128() -> torch.Tensor:
    hadamard = torch.ones((1, 1), dtype=torch.float32, device=DEVICE)
    while hadamard.shape[0] < 128:
        hadamard = torch.cat(
            (torch.cat((hadamard, hadamard), dim=1), torch.cat((hadamard, -hadamard), dim=1)),
            dim=0,
        )
    return hadamard / math.sqrt(128)


def _dense_weight(trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor) -> torch.Tensor:
    """fp32 dense equivalent of one EXL3 tensor: had_128 @ W @ had_128 with the sign flips folded in."""
    size_k, size_n = trellis.shape[0] * 16, trellis.shape[1] * 16
    weight = torch.empty((size_k, size_n), dtype=torch.float16, device=DEVICE)
    ops.exl3_reconstruct(weight, trellis, K_BITS, True, False)
    hadamard = _sylvester_128()
    out = weight.float().view(size_k // 128, 128, size_n)
    out = torch.einsum("ij,bjn->bin", hadamard, out).reshape(size_k, size_n)
    out = out.view(size_k, size_n // 128, 128)
    out = torch.einsum("bki,ij->bkj", out.transpose(0, 1), hadamard).transpose(0, 1).reshape(size_k, size_n)
    return (out * suh.float()[:, None] * svh.float()[None, :]).half().float()


def _loaded_moe(num_experts: int, top_k: int, hidden: int, intermediate: int):
    """RoutedExperts stand-in with synthetic EXL3 expert weights, prepared by ``Exl3MoEMethod``."""
    from aphrodite.model_executor.layers.fused_moe.activation import MoEActivation
    from aphrodite.model_executor.layers.quantization import exl3

    layer = torch.nn.Module()
    layer.layer_name = "test.experts"
    layer.local_num_experts = num_experts
    layer.top_k = top_k
    layer.hidden_size = hidden
    layer.exl3_hidden_size = hidden
    layer.exl3_intermediate_size_per_partition = intermediate
    layer.exl3_tp_rank = 0
    layer.exl3_tp_size = 1
    layer.activation = MoEActivation.SILU
    layer.expert_map = None
    layer.apply_router_weight_on_input = False
    for prefix in ("w13", "w2"):
        for attr in ("suh", "svh", "trellis", "mcg", "mul1"):
            setattr(layer, f"{prefix}_{attr}", SimpleNamespace(exl3_tensors={}))
    mcg = torch.ones(1, dtype=torch.int16, device=DEVICE)
    for expert_id in range(num_experts):
        for shard_id in ("w1", "w3"):
            layer.w13_trellis.exl3_tensors[(expert_id, shard_id)] = _trellis(hidden, intermediate)
            layer.w13_suh.exl3_tensors[(expert_id, shard_id)] = _signs(hidden)
            layer.w13_svh.exl3_tensors[(expert_id, shard_id)] = _signs(intermediate)
            layer.w13_mcg.exl3_tensors[(expert_id, shard_id)] = mcg
        layer.w2_trellis.exl3_tensors[(expert_id, "w2")] = _trellis(intermediate, hidden)
        layer.w2_suh.exl3_tensors[(expert_id, "w2")] = _signs(intermediate)
        layer.w2_svh.exl3_tensors[(expert_id, "w2")] = _signs(hidden)
        layer.w2_mcg.exl3_tensors[(expert_id, "w2")] = mcg
    method = exl3.Exl3MoEMethod.__new__(exl3.Exl3MoEMethod)
    method.quant_config = exl3.Exl3Config()
    method.process_weights_after_loading(layer)
    return layer, method


def _dense_reference(layer, x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor:
    out = torch.zeros((x.shape[0], layer.hidden_size), dtype=torch.float32, device=DEVICE)
    for row in range(x.shape[0]):
        for slot in range(layer.top_k):
            expert_id = int(topk_ids[row, slot])
            w1 = _dense_weight(
                *(getattr(layer, f"w13_{a}").exl3_tensors[(expert_id, "w1")] for a in ("trellis", "suh", "svh"))
            )
            w3 = _dense_weight(
                *(getattr(layer, f"w13_{a}").exl3_tensors[(expert_id, "w3")] for a in ("trellis", "suh", "svh"))
            )
            w2 = _dense_weight(
                *(getattr(layer, f"w2_{a}").exl3_tensors[(expert_id, "w2")] for a in ("trellis", "suh", "svh"))
            )
            act = (torch.nn.functional.silu(x[row].float() @ w1) * (x[row].float() @ w3)).half().float()
            out[row] += float(topk_weights[row, slot]) * (act @ w2)
    return out


@pytest.mark.parametrize(("rows", "intermediate"), [(1, 384), (3, 640)])
def test_moe_small_batch_first_call_matches_dense_reference(rows: int, intermediate: int, monkeypatch, tmp_path):
    """rows=1 takes ``_apply_single_token``, rows=3 ``_apply_small_batch``; both use exl3_mgemm.

    The first ``apply`` below must be the call that autotunes the down
    projection, and it must already be correct. The autotune key covers the
    weight shape, so each case uses an intermediate size no other case (or
    test) shares. The on-disk autotune cache is read once per process; the
    environment override only helps when this test runs first, and is set so
    that running the file on its own never sees a pre-seeded shape.
    """
    monkeypatch.setenv("EXLLAMAV3_TUNE_CACHE", str(tmp_path / "coop_autotune_v1.bin"))
    torch.manual_seed(11)
    layer, method = _loaded_moe(num_experts=16, top_k=4, hidden=256, intermediate=intermediate)
    x = torch.randn((rows, layer.hidden_size), dtype=torch.float16, device=DEVICE) * 0.05
    topk_ids = torch.stack([torch.randperm(layer.local_num_experts, device=DEVICE)[: layer.top_k] for _ in range(rows)])
    topk_weights = torch.softmax(torch.randn((rows, layer.top_k), device=DEVICE), dim=-1)
    reference = _dense_reference(layer, x, topk_ids, topk_weights)

    first = method.apply(layer, x, topk_weights, topk_ids, None, None).float().clone()
    second = method.apply(layer, x, topk_weights, topk_ids, None, None).float()
    torch.cuda.synchronize()

    scale = reference.abs().max().clamp_min(1e-6)
    assert ((first - reference).abs().max() / scale).item() < 1e-2
    torch.testing.assert_close(first, second)
