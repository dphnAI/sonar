# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 output shards must match the unsharded layer at any column boundary.

EXL3 transforms outputs in 128-column Hadamard blocks. A tensor-parallel shard
boundary inside a block (e.g. Qwen3's 151936-entry lm_head split over two ranks)
used to fail at both prefill and decode. Layers are built as rank ``r`` of a
simulated TP group by patching the TP rank/size lookups, so one process checks
every rank against the unsharded layer.
"""

from types import SimpleNamespace

import pytest
import torch

import sonar.model_executor.layers.linear as linear_module
import sonar.model_executor.layers.quantization.exl3 as exl3_module
import sonar.model_executor.parameter as parameter_module
from sonar.config.lora import LoRAConfig
from sonar.lora.layers import LoRAMapping, MergedColumnParallelLinearWithLoRA
from sonar.lora.punica_wrapper import get_punica_wrapper
from sonar.model_executor.layers.linear import MergedColumnParallelLinear, QKVParallelLinear
from sonar.model_executor.layers.quantization.exl3 import Exl3Config, Exl3LinearMethod, _exl3_linear_one
from sonar.platforms import current_platform

if not current_platform.is_cuda() or not current_platform.has_device_capability(80):
    pytest.skip("EXL3 requires CUDA SM80+", allow_module_level=True)

IN_FEATURES = 256
OUT_FEATURES = 640  # five 128-column blocks
BITS = 4


def _tensors(store):
    return SimpleNamespace(exl3_tensors=store)


def _shard_layer(full: dict[str, torch.Tensor], start: int, size: int):
    layer = SimpleNamespace(
        suh=_tensors({None: full["suh"]}),
        svh=_tensors({"full": full["svh"]}),
        trellis=_tensors({"full": full["trellis"]}),
        mcg=_tensors({}),
        mul1=_tensors({}),
        exl3_output_windows={},
    )
    Exl3LinearMethod._slice_output_shard(layer, "full", None, start, size)
    return layer


@pytest.mark.parametrize("tokens", [1, 160], ids=["gemm", "reconstruct"])
@pytest.mark.parametrize(
    "partition",
    [128, 320, 336, 224],
    ids=["aligned", "mid-block", "padded-vocab", "four-ranks-padded"],
)
def test_output_shards_match_unsharded_layer(tokens: int, partition: int):
    torch.manual_seed(0)
    device = torch.device("cuda")
    full = {
        "trellis": torch.randint(
            -(2**15), 2**15, (IN_FEATURES // 16, OUT_FEATURES // 16, 16 * BITS), dtype=torch.int16, device=device
        ),
        "suh": torch.randn(IN_FEATURES, dtype=torch.float16, device=device),
        "svh": torch.randn(OUT_FEATURES, dtype=torch.float16, device=device),
    }
    x = torch.randn(tokens, IN_FEATURES, dtype=torch.float16, device=device)
    expected = _exl3_linear_one(x, full["trellis"], full["suh"], full["svh"], False, False)

    method = Exl3LinearMethod.__new__(Exl3LinearMethod)
    ranks = -(-OUT_FEATURES // partition)
    outputs = [method._apply_one(_shard_layer(full, rank * partition, partition), x, None) for rank in range(ranks)]
    actual = torch.cat(outputs, dim=-1)

    assert actual.shape == (tokens, ranks * partition)
    torch.testing.assert_close(actual[:, :OUT_FEATURES], expected, rtol=1e-2, atol=1e-2)
    # Vocab padding past the checkpoint is zero.
    assert torch.count_nonzero(actual[:, OUT_FEATURES:]) == 0


def _random_exl3(out_features: int, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return {
        "trellis": torch.randint(
            -(2**15),
            2**15,
            (IN_FEATURES // 16, out_features // 16, 16 * BITS),
            dtype=torch.int16,
            device="cuda",
            generator=generator,
        ),
        "suh": torch.randn(IN_FEATURES, dtype=torch.float16, device="cuda", generator=generator),
        "svh": torch.randn(out_features, dtype=torch.float16, device="cuda", generator=generator),
    }


def _full_output(x: torch.Tensor, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    return _exl3_linear_one(x, tensors["trellis"], tensors["suh"], tensors["svh"], False, False)


def _build_rank_layer(monkeypatch, kind: str, rank: int, tp_size: int, dims: dict[str, int], full):
    """Build and load a real EXL3 merged layer as ``rank`` of ``tp_size``."""
    for module in (linear_module, exl3_module, parameter_module):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: rank)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: tp_size)
    projections = ("gate_proj", "up_proj") if kind == "gate_up" else ("q_proj", "k_proj", "v_proj")
    config = Exl3Config(tensor_storage={f"model.layers.0.{kind}.{p}": {"quant_format": "exl3"} for p in projections})
    if kind == "gate_up":
        layer = MergedColumnParallelLinear(
            IN_FEATURES,
            [dims["intermediate"]] * 2,
            bias=False,
            params_dtype=torch.float16,
            quant_config=config,
            prefix="model.layers.0.gate_up.gate_up_proj",
        )
    else:
        layer = QKVParallelLinear(
            IN_FEATURES,
            dims["head_size"],
            dims["num_heads"],
            dims["num_kv_heads"],
            bias=False,
            params_dtype=torch.float16,
            quant_config=config,
            prefix="model.layers.0.qkv.qkv_proj",
        )
    assert isinstance(layer.quant_method, Exl3LinearMethod)
    for shard_id in layer.exl3_shard_ids:
        for attr in ("suh", "svh", "trellis"):
            getattr(layer, attr).load_exl3_weight(full[shard_id][attr], shard_id)
    layer.quant_method.process_weights_after_loading(layer)
    return layer


def _expected_rank_output(x: torch.Tensor, full, layer, rank: int) -> torch.Tensor:
    pieces = []
    for shard_id, size in zip(layer.exl3_shard_ids, layer.exl3_output_partition_sizes):
        shard_rank = rank // layer.num_kv_head_replicas if shard_id in ("k", "v") else rank
        pieces.append(_full_output(x, full[shard_id])[:, shard_rank * size : (shard_rank + 1) * size])
    return torch.cat(pieces, dim=-1)


MERGED_CASES = {
    # Each rank's slice of every projection is a whole number of 128-column blocks.
    "gate_up-aligned": ("gate_up", {"intermediate": 512}, True),
    "gate_up-mid-block": ("gate_up", {"intermediate": 640}, False),
    "qkv-aligned": ("qkv", {"head_size": 128, "num_heads": 4, "num_kv_heads": 2}, True),
    # Full widths stay 128-multiples (640/128); per-rank q (320) and k/v (64) cut blocks.
    "qkv-mid-block": ("qkv", {"head_size": 64, "num_heads": 10, "num_kv_heads": 2}, False),
}


def _full_tensors(kind: str, dims: dict[str, int]):
    if kind == "gate_up":
        return {0: _random_exl3(dims["intermediate"], 1), 1: _random_exl3(dims["intermediate"], 2)}
    q = dims["head_size"] * dims["num_heads"]
    kv = dims["head_size"] * dims["num_kv_heads"]
    return {"q": _random_exl3(q, 1), "k": _random_exl3(kv, 2), "v": _random_exl3(kv, 3)}


@pytest.mark.parametrize("tokens", [1, 160], ids=["gemm", "reconstruct"])
@pytest.mark.parametrize("case", list(MERGED_CASES))
def test_merged_layer_ranks_match_unsharded(default_sonar_config, monkeypatch, case: str, tokens: int):
    """Fused qkv/gate_up kernels are kept for aligned shards and bypassed otherwise."""
    torch.manual_seed(0)
    kind, dims, fused = MERGED_CASES[case]
    full = _full_tensors(kind, dims)
    x = torch.randn(tokens, IN_FEATURES, dtype=torch.float16, device="cuda")
    tp_size = 2
    for rank in range(tp_size):
        layer = _build_rank_layer(monkeypatch, kind, rank, tp_size, dims, full)
        assert layer.exl3_can_mgemm is fused
        actual, _ = layer(x)
        # atol covers fp16 rounding between the fused and single kernels; a shard
        # cut inside a Hadamard block is off by O(1).
        torch.testing.assert_close(actual, _expected_rank_output(x, full, layer, rank), rtol=1e-2, atol=5e-2)


@torch.inference_mode()
@pytest.mark.parametrize("tokens", [1, 160], ids=["gemm", "reconstruct"])
@pytest.mark.parametrize("case", ["gate_up-aligned", "gate_up-mid-block"])
def test_lora_on_sharded_exl3_matches_unsharded(default_sonar_config, monkeypatch, case: str, tokens: int):
    """A gate_up LoRA adds to the right columns of every EXL3 rank shard."""
    torch.manual_seed(0)
    kind, dims, _ = MERGED_CASES[case]
    full = _full_tensors(kind, dims)
    rank_dim, intermediate = 8, dims["intermediate"]
    lora_config = LoRAConfig(max_loras=1, max_lora_rank=rank_dim, lora_dtype=torch.float16)
    punica_wrapper = get_punica_wrapper(8192, 256, torch.device("cuda"), lora_config=lora_config)
    punica_wrapper.update_metadata(LoRAMapping((1,) * tokens, (1,) * tokens, is_prefill=True), [1], 1, 512)
    lora_a = [torch.randn(rank_dim, IN_FEATURES, dtype=torch.float16, device="cuda") * 0.1 for _ in range(2)]
    lora_b = [torch.randn(intermediate, rank_dim, dtype=torch.float16, device="cuda") * 0.1 for _ in range(2)]
    x = torch.randn(tokens, IN_FEATURES, dtype=torch.float16, device="cuda")

    tp_size = 2
    for rank in range(tp_size):
        layer = _build_rank_layer(monkeypatch, kind, rank, tp_size, dims, full)
        lora_layer = MergedColumnParallelLinearWithLoRA(layer)
        lora_layer.create_lora_weights(1, lora_config)
        lora_layer.set_mapping(punica_wrapper)
        lora_layer.set_lora(0, lora_a, lora_b)

        actual, _ = lora_layer(x)
        base = _expected_rank_output(x, full, layer, rank)
        size = intermediate // tp_size
        columns = slice(rank * size, (rank + 1) * size)
        delta = torch.cat([x.float() @ a.float().T @ b[columns].float().T for a, b in zip(lora_a, lora_b)], dim=-1)
        assert delta.abs().max() > 0.5
        torch.testing.assert_close(actual.float(), base.float() + delta, rtol=1e-2, atol=5e-2)
