# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial
from types import SimpleNamespace

import pytest
import ray
import torch
import torch.distributed as dist
import torch.nn.functional as F

from sonar.distributed import get_tp_group
from sonar.model_executor.layers.fused_moe.config import FusedMoEConfig
from sonar.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe import (
    TrtLlmMxfp4ExpertsMonolithic,
)
from sonar.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from sonar.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from sonar.model_executor.layers.linear import UnquantizedLinearMethod
from sonar.model_executor.warmup.cutedsl_warmup import cutedsl_warmup
from sonar.models.kimi_k3.nvidia import latent_moe_runner
from sonar.models.kimi_k3.nvidia.ops.latent_moe_tail import KimiK3LatentMoETailOp
from sonar.platforms import current_platform
from tests.utils import (
    init_test_distributed_environment,
    multi_gpu_test,
    multi_process_parallel,
)

HIDDEN_SIZE = 7168
LATENT_SIZE = 3584
EPS = 0.1
TOP_K = 8


def test_deferred_finalize_enabled_before_moe_kernel_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeMoEConfig:
        tp_size = 8
        dp_size = 1
        ep_size = 1
        pcp_size = 1
        is_sequence_parallel = False
        hidden_dim = LATENT_SIZE
        hidden_dim_unpadded = LATENT_SIZE
        experts_per_token = 16
        _defer_moe_finalize = False
        defer_moe_finalize_max_num_tokens = -1
        defer_moe_finalize = FusedMoEConfig.defer_moe_finalize
        limit_deferred_moe_finalize = FusedMoEConfig.limit_deferred_moe_finalize
        use_deferred_moe_finalize = FusedMoEConfig.use_deferred_moe_finalize

    moe_config = FakeMoEConfig()
    quant_method = SimpleNamespace(
        experts_cls=TrtLlmMxfp4ExpertsMonolithic,
        moe_kernel=None,
    )
    norm_weight = torch.empty(LATENT_SIZE, dtype=torch.bfloat16)
    transform = SimpleNamespace(
        norm=SimpleNamespace(weight=norm_weight, variance_epsilon=EPS),
        up_proj=SimpleNamespace(
            weight=SimpleNamespace(shape=(HIDDEN_SIZE, LATENT_SIZE)),
            quant_method=UnquantizedLinearMethod(),
        ),
    )

    def fake_runner_init(runner, *args, **kwargs) -> None:
        runner.moe_config = moe_config
        runner.routed_experts = SimpleNamespace(quant_method=quant_method)
        runner._shared_experts = object()
        runner.routed_output_transform = transform

    initialized_with: dict[str, object] = {}
    tail_op = SimpleNamespace(contract=SimpleNamespace(max_num_tokens=128))

    def fake_tail_initialize(**kwargs):
        initialized_with.update(kwargs)
        return tail_op

    monkeypatch.setattr(MoERunner, "__init__", fake_runner_init)
    monkeypatch.setattr(latent_moe_runner.torch.cuda, "Event", lambda: object())
    monkeypatch.setattr(
        latent_moe_runner,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: True,
            is_device_capability_family=lambda capability: capability == 100,
            current_device=lambda: torch.device("cuda"),
        ),
    )
    monkeypatch.setattr(
        latent_moe_runner,
        "get_current_sonar_config",
        lambda: SimpleNamespace(
            parallel_config=SimpleNamespace(use_ubatching=False),
            model_config=SimpleNamespace(enable_sleep_mode=False),
        ),
    )
    monkeypatch.setattr(KimiK3LatentMoETailOp, "initialize", fake_tail_initialize)

    latent_moe_runner.LatentMoERunner()

    # The stand-in deliberately borrows the real config's property.
    assert moe_config.use_deferred_moe_finalize  # type: ignore[misc]
    assert moe_config.defer_moe_finalize_max_num_tokens == 128
    assert initialized_with["experts_per_token"] == 16


def _make_deferred_routed_output(
    num_tokens: int,
    device: torch.device,
) -> tuple[UnfinalizedMoEOutput, torch.Tensor]:
    num_routes = num_tokens * TOP_K
    num_permuted_rows = num_routes + 7
    expanded_output = torch.randn(
        num_routes,
        LATENT_SIZE,
        device=device,
        dtype=torch.bfloat16,
    ).mul_(0.01)
    expert_weights = torch.rand(
        num_tokens,
        TOP_K,
        device=device,
        dtype=torch.bfloat16,
    )
    expert_weights.div_(expert_weights.sum(dim=-1, keepdim=True))
    expanded_idx = torch.randperm(num_permuted_rows, device=device)[:num_routes]
    gemm2_permuted = torch.empty(
        num_permuted_rows,
        LATENT_SIZE,
        device=device,
        dtype=torch.bfloat16,
    )
    gemm2_permuted[expanded_idx] = expanded_output

    finalized = torch.zeros(
        num_tokens,
        LATENT_SIZE,
        device=device,
        dtype=torch.bfloat16,
    )
    expanded_output = expanded_output.view(num_tokens, TOP_K, LATENT_SIZE)
    for slot in range(TOP_K):
        finalized.add_(expanded_output[:, slot] * expert_weights[:, slot, None])

    return (
        UnfinalizedMoEOutput(
            gemm2_permuted=gemm2_permuted,
            expert_weights=expert_weights,
            expanded_idx_to_permuted_idx=expanded_idx.to(torch.int32).view(num_tokens, TOP_K),
        ),
        finalized,
    )


def _make_bf16_top16_deferred_output(
    num_tokens: int,
    device: torch.device,
    *,
    drop_last_route: bool = False,
) -> tuple[UnfinalizedMoEOutput, torch.Tensor]:
    top_k = 16
    num_routes = num_tokens * top_k
    num_permuted_rows = num_routes + 7
    expanded_output = torch.randn(
        num_routes,
        LATENT_SIZE,
        device=device,
        dtype=torch.bfloat16,
    ).mul_(0.01)
    expert_weights = torch.rand(
        num_tokens,
        top_k,
        device=device,
        dtype=torch.bfloat16,
    )
    expert_weights.div_(expert_weights.sum(dim=-1, keepdim=True))
    expanded_idx = torch.randperm(num_permuted_rows, device=device)[:num_routes]
    gemm2_permuted = torch.empty(
        num_permuted_rows,
        LATENT_SIZE,
        device=device,
        dtype=torch.bfloat16,
    )
    gemm2_permuted[expanded_idx] = expanded_output

    expanded_output = expanded_output.view(num_tokens, top_k, LATENT_SIZE)
    expanded_idx = expanded_idx.view(num_tokens, top_k)
    if drop_last_route:
        expanded_output[:, -1].zero_()
        expanded_idx[:, -1] = -1
    finalized = (expanded_output.float() * expert_weights[:, :, None].float()).sum(1).to(torch.bfloat16)
    return (
        UnfinalizedMoEOutput(
            gemm2_permuted=gemm2_permuted,
            expert_weights=expert_weights,
            expanded_idx_to_permuted_idx=expanded_idx.to(torch.int32),
        ),
        finalized,
    )


@pytest.mark.parametrize("shard_dim", [448, 896])
@pytest.mark.parametrize(("base", "with_escapes"), [(0, False), (60, True), (120, True)])
def test_nf12_skinny_register_decode(shard_dim: int, base: int, with_escapes: bool) -> None:
    """Check recovered representations before FMA can hide slot/NaN errors."""
    if not current_platform.is_device_capability_family(100):
        pytest.skip("K3 latent-MoE tail fusion requires SM100")
    pytest.importorskip("nf12")
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from nf12 import reference

    from sonar.models.kimi_k3.nvidia.ops.cute_dsl.latent_moe_tail.fused_add_multicast_skinny_gemm import (
        FusedAddMulticastSkinnyGemm,
        _as_cute,
        config_for_m,
    )

    # Prefix rows make escape IDs non-local. Alternating signs avoid sentinel
    # collisions in the ordinary data, including the empty-escape-table case.
    index = torch.arange((shard_dim + 16) * LATENT_SIZE, dtype=torch.int32)
    row = index // LATENT_SIZE
    bits = (((index + row * 3) & 15) + 2 * base) << 7 | (((index >> 4) + row * 7) & 127) | ((index & 1) << 15)
    weight = bits.to(torch.int16).view(shard_dim + 16, LATENT_SIZE).view(torch.bfloat16)
    if with_escapes:
        weight[:16].zero_()
        # Every BF16 representation, including signed zeros, subnormals and NaNs.
        weight[16:48].view(torch.int16).copy_(
            (torch.arange(32 * LATENT_SIZE, dtype=torch.int32) & 0xFFFF).to(torch.int16).view(32, LATENT_SIZE)
        )
        # In-window positive groups whose meta word collides with the sentinel.
        weight[48:64].view(torch.int16).fill_(base << 8)
    packed, escapes, _ = reference.encode(weight, exp_base=base)
    assert (escapes.shape[0] > 0) == with_escapes
    packed = packed[16:].cuda()
    escapes = escapes.cuda()
    decoded = torch.empty((shard_dim, LATENT_SIZE), dtype=torch.bfloat16, device="cuda")

    class DecodeRegisters:
        def __init__(self):
            self.loader = FusedAddMulticastSkinnyGemm(
                num_rows=1, hidden_dim=HIDDEN_SIZE, config=config_for_m(1, shard_dim, nf12=True), nf12=True
            )

        @cute.jit
        def __call__(
            self,
            packed: cute.Tensor,
            output: cute.Tensor,
            escapes_ptr: cutlass.Int64,
            exp_base: cutlass.Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(packed, output, escapes_ptr, exp_base).launch(
                grid=[shard_dim // 4, 1, 1],
                block=[224, 1, 1],
                smem=self.loader.shared_bytes,
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            packed: cute.Tensor,
            output: cute.Tensor,
            escapes_ptr: cutlass.Int64,
            exp_base: cutlass.Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            block_idx, _, _ = cute.arch.block_idx()
            registers = cute.make_rmem_tensor(cute.make_layout((4, 16), stride=(16, 1)), cutlass.BFloat16)
            n_base = (block_idx // 4) * 16 + (block_idx % 4) * 2
            self.loader.load_nf12(packed, registers, n_base, tidx, cutlass.Int32(LATENT_SIZE), escapes_ptr, exp_base)
            for ni in cutlass.range_constexpr(4):
                # Reassemble the logical row order, not the CTA register order.
                output_row = n_base + ni % 2 + (ni // 2) * 8
                for vi in cutlass.range_constexpr(16):
                    output[output_row, tidx * 16 + vi] = registers[ni, vi]

    args = (
        _as_cute(packed.view(torch.uint32)),
        _as_cute(decoded),
        cutlass.Int64(escapes.data_ptr()),
        cutlass.Int32(base),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    compiled = cute.compile(DecodeRegisters(), *args)
    compiled(*args)
    assert torch.equal(decoded.cpu().view(torch.int16), weight[16:].view(torch.int16))


@ray.remote(num_gpus=1, max_calls=1)
def _test_latent_moe_tail_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
    use_nf12: bool = False,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(
        tp_size,
        pp_size,
        rank,
        distributed_init_port,
    )

    torch.manual_seed(0)
    rms_weight = 1 + 0.1 * torch.randn(
        LATENT_SIZE,
        device=device,
        dtype=torch.bfloat16,
    )
    up_weight = (
        torch.randn(
            HIDDEN_SIZE,
            LATENT_SIZE,
            device=device,
            dtype=torch.bfloat16,
        )
        / LATENT_SIZE**0.5
    )
    shard_size = HIDDEN_SIZE // tp_size
    # Escapes span both row halves, all four atoms, and the final K tile.
    for row in (0, 7, 8, 15):
        for col in (0, 7, 8, 15, 16, 31, 32, 47, 48, 63, LATENT_SIZE - 1):
            up_weight[row::16, col] = -1e-20 if (row + col) % 2 else 1e-20
    local_up_weight = up_weight.narrow(0, rank * shard_size, shard_size)
    weight_metadata = {}
    if use_nf12:
        import nf12

        from sonar.model_executor.layers.quantization.online.nf12 import Nf12OnlineLinearMethod

        encoded = nf12.encode(up_weight)
        method = object.__new__(Nf12OnlineLinearMethod)
        method.fallback = False
        runner = SimpleNamespace(
            moe_config=SimpleNamespace(tp_size=tp_size),
            routed_output_transform=SimpleNamespace(
                up_proj=SimpleNamespace(
                    weight=encoded.packed,
                    weight_escapes=encoded.escapes,
                    weight_exp_base=encoded.exp_base,
                    quant_method=method,
                )
            ),
        )
        decoded_local_weight = latent_moe_runner.LatentMoERunner._up_proj_weight_shard(runner)
        assert torch.equal(decoded_local_weight.view(torch.int16), local_up_weight.view(torch.int16))
        get_local_weight = partial(latent_moe_runner.LatentMoERunner._up_proj_weight_shard, runner, decode_nf12=False)
        weight_metadata = dict(weight_escapes=encoded.escapes, weight_exp_base=encoded.exp_base)
    else:
        get_local_weight = partial(up_weight.narrow, 0, rank * shard_size, shard_size)

    group = get_tp_group().device_group
    op = KimiK3LatentMoETailOp.initialize(
        hidden_size=HIDDEN_SIZE,
        latent_size=LATENT_SIZE,
        dtype=torch.bfloat16,
        device=device,
        rms_eps=EPS,
        nf12_weights=use_nf12,
    )
    cutedsl_warmup()

    # M=65 crosses two 32-CTA token waves and reuses the first DSM slot.
    for iteration, num_tokens in enumerate((1, 2, 3, 4, 5, 6, 7, 8, 16, 33, 65, 5)):
        torch.manual_seed(100 * iteration + rank + 1)
        routed_output = torch.randn(
            num_tokens,
            LATENT_SIZE,
            device=device,
            dtype=torch.bfloat16,
        ).mul_(0.01)
        shared_output = torch.randn(
            num_tokens,
            HIDDEN_SIZE,
            device=device,
            dtype=torch.bfloat16,
        )
        # Distinguish both row halves in each 16-row tile's shared-add epilogue.
        shared_output.add_((torch.arange(HIDDEN_SIZE, device=device) % 16).to(torch.bfloat16).mul_(0.125))

        routed_reference = routed_output.clone()
        shared_reference = shared_output.clone()
        dist.all_reduce(routed_reference, group=group)
        dist.all_reduce(shared_reference, group=group)
        expected = F.linear(
            F.rms_norm(
                routed_reference,
                (LATENT_SIZE,),
                rms_weight,
                EPS,
            ),
            up_weight,
        )
        expected.add_(shared_reference)

        actual = op(
            routed_output,
            shared_output,
            rms_weight,
            get_local_weight(),
            **weight_metadata,
        )
        torch.testing.assert_close(actual, expected, atol=8e-2, rtol=3e-2)
        assert actual.is_contiguous()

        if use_nf12:
            # Check projection rounding, shared-row mapping, and multicast packets.
            packed_result = actual.clone()
            dense_result = op(routed_output, shared_output, rms_weight, local_up_weight).clone()
            assert torch.equal(packed_result.view(torch.int16), dense_result.view(torch.int16)), (
                f"M={num_tokens}, iteration={iteration}, eager"
            )

        if num_tokens in (1, 5, 6, 7, 8):
            alternate_weight = nf12.encode(-up_weight) if use_nf12 and num_tokens == 8 else None
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_output = op(
                    routed_output,
                    shared_output,
                    rms_weight,
                    get_local_weight(),
                    **weight_metadata,
                ).clone()
                if alternate_weight is not None:
                    # Consecutive layers may reuse the transient decode allocation.
                    alternate_output = op(
                        routed_output,
                        shared_output,
                        rms_weight,
                        alternate_weight.packed.narrow(0, rank * shard_size, shard_size),
                        weight_escapes=alternate_weight.escapes,
                        weight_exp_base=alternate_weight.exp_base,
                    ).clone()
            for replay in range(3):
                # Reuse graph pointers with changed inputs and mailbox generations.
                shared_output.add_(0.125)
                dense_result = op(routed_output, shared_output, rms_weight, local_up_weight).clone()
                graph.replay()
                graph_result = graph_output.clone()
                eager_result = op(
                    routed_output,
                    shared_output,
                    rms_weight,
                    get_local_weight(),
                    **weight_metadata,
                ).clone()
                assert torch.equal(graph_result.view(torch.int16), eager_result.view(torch.int16)), (
                    f"M={num_tokens}, iteration={iteration}, replay={replay}, graph differs from eager"
                )
                assert torch.equal(graph_result.view(torch.int16), dense_result.view(torch.int16)), (
                    f"M={num_tokens}, iteration={iteration}, replay={replay}, "
                    f"max_error={(graph_result.float() - dense_result.float()).abs().max().item()}"
                )
                if alternate_weight is not None:
                    alternate_reference = op(routed_output, shared_output, rms_weight, -local_up_weight).clone()
                    assert torch.equal(alternate_output.view(torch.int16), alternate_reference.view(torch.int16)), (
                        f"M={num_tokens}, replay={replay}, consecutive weight decode"
                    )


def _run_latent_moe_tail_test(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    use_nf12: bool = False,
) -> None:
    if not current_platform.is_device_capability_family(100):
        pytest.skip("K3 latent-MoE tail fusion requires SM100")
    if use_nf12:
        pytest.importorskip("nf12")
    multi_process_parallel(
        monkeypatch,
        tp_size,
        1,
        SimpleNamespace(remote=partial(_test_latent_moe_tail_worker.remote, use_nf12=use_nf12)),
    )


@ray.remote(num_gpus=1, max_calls=1)
def _test_deferred_finalize_parity_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(
        tp_size,
        pp_size,
        rank,
        distributed_init_port,
    )

    torch.manual_seed(1000 + rank)
    rms_weight = 1 + 0.1 * torch.randn(
        LATENT_SIZE,
        device=device,
        dtype=torch.bfloat16,
    )
    up_weight = (
        torch.randn(
            HIDDEN_SIZE,
            LATENT_SIZE,
            device=device,
            dtype=torch.bfloat16,
        )
        / LATENT_SIZE**0.5
    )
    local_up_weight = up_weight.narrow(0, rank * (HIDDEN_SIZE // tp_size), HIDDEN_SIZE // tp_size)
    finalized_op = KimiK3LatentMoETailOp.initialize(
        hidden_size=HIDDEN_SIZE,
        latent_size=LATENT_SIZE,
        dtype=torch.bfloat16,
        device=device,
        rms_eps=EPS,
    )
    deferred_op = KimiK3LatentMoETailOp.initialize(
        hidden_size=HIDDEN_SIZE,
        latent_size=LATENT_SIZE,
        dtype=torch.bfloat16,
        device=device,
        rms_eps=EPS,
        experts_per_token=TOP_K,
    )
    bf16_deferred_op = KimiK3LatentMoETailOp.initialize(
        hidden_size=HIDDEN_SIZE,
        latent_size=LATENT_SIZE,
        dtype=torch.bfloat16,
        device=device,
        rms_eps=EPS,
        experts_per_token=16,
    )
    cutedsl_warmup()

    for iteration, num_tokens in enumerate((1, 5, 16, 33, 65)):
        torch.manual_seed(2000 + 100 * iteration + rank)
        deferred_output, finalized_output = _make_deferred_routed_output(
            num_tokens,
            device,
        )
        shared_output = torch.randn(
            num_tokens,
            HIDDEN_SIZE,
            device=device,
            dtype=torch.bfloat16,
        )

        expected = finalized_op(
            finalized_output,
            shared_output,
            rms_weight,
            local_up_weight,
        )
        actual = deferred_op(
            deferred_output,
            shared_output,
            rms_weight,
            local_up_weight,
        )

        torch.testing.assert_close(actual, expected, atol=8e-2, rtol=3e-2)

    for iteration, num_tokens in enumerate((1, 2, 4, 5, 16)):
        torch.manual_seed(3000 + 100 * iteration + rank)
        deferred_output, finalized_output = _make_bf16_top16_deferred_output(
            num_tokens,
            device,
            drop_last_route=num_tokens == 5,
        )
        shared_output = torch.randn(
            num_tokens,
            HIDDEN_SIZE,
            device=device,
            dtype=torch.bfloat16,
        )

        expected = finalized_op(
            finalized_output,
            shared_output,
            rms_weight,
            local_up_weight,
        )
        actual = bf16_deferred_op(
            deferred_output,
            shared_output,
            rms_weight,
            local_up_weight,
        )

        torch.testing.assert_close(actual, expected, atol=8e-2, rtol=3e-2)
        if num_tokens == 4:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_actual = bf16_deferred_op(
                    deferred_output,
                    shared_output,
                    rms_weight,
                    local_up_weight,
                )
            graph.replay()
            torch.testing.assert_close(
                graph_actual,
                expected,
                atol=8e-2,
                rtol=3e-2,
            )


def _run_deferred_finalize_parity_test(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
) -> None:
    if not current_platform.is_device_capability_family(100):
        pytest.skip("K3 latent-MoE tail fusion requires SM100")
    multi_process_parallel(
        monkeypatch,
        tp_size,
        1,
        _test_deferred_finalize_parity_worker,
    )


@multi_gpu_test(num_gpus=8)
@pytest.mark.parametrize("use_nf12", [False, True])
def test_latent_moe_tail_tp8_matches_native_path(
    monkeypatch: pytest.MonkeyPatch,
    use_nf12: bool,
) -> None:
    _run_latent_moe_tail_test(monkeypatch, 8, use_nf12)


@multi_gpu_test(num_gpus=16)
@pytest.mark.parametrize("use_nf12", [False, True])
def test_latent_moe_tail_tp16_matches_native_path(
    monkeypatch: pytest.MonkeyPatch,
    use_nf12: bool,
) -> None:
    _run_latent_moe_tail_test(monkeypatch, 16, use_nf12)


@multi_gpu_test(num_gpus=8)
def test_latent_moe_tail_deferred_finalize_matches_finalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_deferred_finalize_parity_test(monkeypatch, 8)
