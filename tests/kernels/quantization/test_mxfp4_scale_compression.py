# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from sonar.model_executor.layers.quantization.utils.mxfp4_scale_compression import (
    encode_mxfp4_scales,
)
from sonar.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")


def _scales(width: int, distribution: str, dtype: torch.dtype) -> torch.Tensor:
    indices = torch.arange(5 * width, device="cuda")
    common = 120 + indices % 3
    if distribution == "all_codes":
        values = indices % 256
    elif distribution == "mixed":
        values = torch.where(indices % 127 == 0, indices % 256, common)
    else:
        values = common
    return values.to(torch.uint8).reshape(5, width).view(dtype)


@pytest.mark.parametrize(
    "n13,n2,dtype,distribution",
    [
        (259, 131, torch.uint8, "all_codes"),
        (256, 128, torch.float8_e4m3fn, "common"),
        (256, 132, torch.float32, "mixed"),
    ],
)
def test_scale_storage_roundtrip(n13, n2, dtype, distribution):
    w13 = _scales(n13, distribution, dtype)
    w2 = _scales(n2, distribution, dtype)
    codec = encode_mxfp4_scales(w13, w2)
    scratch = torch.empty(codec.raw_nbytes, dtype=torch.uint8, device="cuda")
    codec.decode_into(scratch)
    actual13, actual2 = codec.destination_views(scratch)
    assert torch.equal(actual13, w13.view(torch.uint8))
    assert torch.equal(actual2, w2.view(torch.uint8))


@pytest.mark.parametrize(
    "count,dtype,copy_ids",
    [(8, torch.int16, True), (128, torch.int32, True), (8, torch.int16, False)],
)
def test_selected_scales_graph_replay(count, dtype, copy_ids):
    w13 = _scales(259, "mixed", torch.uint8)
    w2 = _scales(131, "all_codes", torch.uint8)
    codec = encode_mxfp4_scales(w13, w2)
    scratch = torch.empty(codec.raw_nbytes, dtype=torch.uint8, device="cuda")
    ids = torch.full((count // 8, 8), 100, dtype=dtype, device="cuda")
    ids_out = torch.empty_like(ids, dtype=torch.int32) if copy_ids else None
    codec.decode_into(scratch, ids, expert_offset=100, expert_ids_out=ids_out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        codec.decode_into(scratch, ids, expert_offset=100, expert_ids_out=ids_out)

    for selected in ((0, 2), (1, 4), (0, 2)):
        a, b = selected
        values = [100 + a, 100 + b, 100 + b, 99, -1, 105, 100 + a, 100 + b]
        ids.copy_(torch.tensor(values, dtype=dtype, device="cuda").expand_as(ids))
        scratch.fill_(19)
        graph.replay()
        actual13, actual2 = codec.destination_views(scratch)
        expected13, expected2 = torch.full_like(w13, 19), torch.full_like(w2, 19)
        for expert in selected:
            expected13[expert] = w13[expert]
            expected2[expert] = w2[expert]
        assert torch.equal(actual13, expected13)
        assert torch.equal(actual2, expected2)
        if ids_out is not None:
            assert torch.equal(ids_out, ids.to(torch.int32))


def test_tuning_initializes_unselected_scales(monkeypatch):
    from flashinfer.autotuner import autotune

    from sonar.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe import (
        TrtLlmMxfp4ExpertsBase,
    )
    from sonar.v1.worker import workspace

    monkeypatch.setattr(workspace, "_manager", workspace.WorkspaceManager(torch.device("cuda")))
    w13 = _scales(259, "mixed", torch.uint8)
    w2 = _scales(131, "all_codes", torch.uint8)
    expert = TrtLlmMxfp4ExpertsBase.__new__(TrtLlmMxfp4ExpertsBase)
    expert._compressed_scales = encode_mxfp4_scales(w13, w2)
    expert._scale_dtypes = (torch.uint8, torch.uint8)
    expert.ep_rank = 0
    expert.local_num_experts = 5
    ids = torch.tensor([[0, 2, 2, -1]], dtype=torch.int16, device="cuda")
    ids_out = torch.empty_like(ids, dtype=torch.int32)
    with autotune(tune_mode=True):
        actual13, actual2 = expert._weight_scales(ids, ids_out)
    assert torch.equal(actual13, w13)
    assert torch.equal(actual2, w2)
    assert torch.equal(ids_out, ids.to(torch.int32))

    actual13.fill_(19)
    actual2.fill_(19)
    actual13, actual2 = expert._weight_scales(ids)
    expected13, expected2 = torch.full_like(w13, 19), torch.full_like(w2, 19)
    for selected in (0, 2):
        expected13[selected] = w13[selected]
        expected2[selected] = w2[selected]
    assert torch.equal(actual13, expected13)
    assert torch.equal(actual2, expected2)


@pytest.mark.parametrize(
    "tokens,defer,logits_dtype,bias_dtype,method",
    [
        (1, False, torch.float32, torch.float32, "DeepSeekV3"),
        (8, True, torch.float32, torch.float32, "DeepSeekV3"),
        (16, True, torch.float32, torch.float32, "DeepSeekV3"),
        (1, True, torch.bfloat16, torch.float32, "DeepSeekV3"),
        (8, False, torch.float32, torch.bfloat16, "DeepSeekV3"),
        (1, False, torch.bfloat16, torch.bfloat16, "DeepSeekV3"),
        (1, False, torch.float32, None, "Renormalize"),
        (8, True, torch.float32, None, "RenormalizeNaive"),
    ],
)
def test_compressed_native_experts(monkeypatch, tokens, defer, logits_dtype, bias_dtype, method):
    if not current_platform.is_device_capability_family(100):
        pytest.skip("Requires SM100-family TRTLLM MXFP4 kernels")
    from flashinfer import mxfp8_quantize
    from flashinfer.autotuner import AutoTuner
    from flashinfer.fused_moe.core import get_trtllm_moe_sm100_module

    from sonar.model_executor.layers.fused_moe.activation import MoEActivation
    from sonar.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
        mxfp4_moe_quant_config,
    )
    from sonar.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe import (
        TrtLlmMxfp4ExpertsMonolithic,
    )
    from sonar.v1.worker import workspace

    monkeypatch.setattr(workspace, "_manager", workspace.WorkspaceManager(torch.device("cuda")))
    monkeypatch.setenv("SONAR_MXFP4_COMPRESS_SCALES", "1")
    e, h, i, k = 128, 512, 256, 16
    parallel = FusedMoEParallelConfig(
        tp_size=2 if defer else 1,
        pcp_size=1,
        dp_size=1,
        ep_size=1,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=1,
        use_ep=False,
        all2all_backend="allgather_reducescatter",
        enable_eplb=False,
    )
    config = FusedMoEConfig(
        num_experts=e,
        experts_per_token=k,
        hidden_dim=h,
        intermediate_size=i * (2 if defer else 1),
        num_local_experts=e,
        num_logical_experts=e,
        activation=MoEActivation.SITU,
        device="cuda",
        routing_method=RoutingMethodType[method],
        moe_parallel_config=parallel,
        in_dtype=torch.bfloat16,
        router_logits_dtype=logits_dtype,
        max_num_tokens=16,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )
    config._defer_moe_finalize = defer
    generator = torch.Generator(device="cuda").manual_seed(302)
    w13 = torch.randint(
        0,
        256,
        (e, 2 * i, h // 2),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    w2 = torch.randint(
        0,
        256,
        (e, h, i // 2),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    scales = []
    for shape in ((e, 2 * i, h // 32), (e, h, i // 32)):
        scale = torch.full(shape, 121, dtype=torch.uint8, device="cuda")
        scale.flatten()[::997] = 119
        scales.append(scale.view(torch.float8_e4m3fn))
    ordinary = TrtLlmMxfp4ExpertsMonolithic(config, mxfp4_moe_quant_config(*scales))
    layer = torch.nn.Module()
    for name, scale in zip(("w13_weight_scale", "w2_weight_scale"), scales):
        layer.register_parameter(name, torch.nn.Parameter(scale.clone(), requires_grad=False))
    compressed = TrtLlmMxfp4ExpertsMonolithic(
        config, mxfp4_moe_quant_config(layer.w13_weight_scale, layer.w2_weight_scale)
    )
    compressed.process_weights_after_loading(layer)
    assert layer.w13_weight_scale is None and layer.w2_weight_scale is None
    assert compressed.quant_config.w1_scale is None
    assert compressed.quant_config.w2_scale is None
    x = torch.randn((tokens, h), dtype=torch.bfloat16, device="cuda", generator=generator) * 0.2
    q, sf = mxfp8_quantize(x, is_sf_swizzled_layout=False)
    sf = sf.view(torch.float8_e4m3fn).reshape(tokens, -1)
    logits = torch.randn((tokens, e), dtype=logits_dtype, device="cuda", generator=generator)
    bias = (
        torch.randn(e, dtype=bias_dtype, device="cuda", generator=generator) * 0.1 if bias_dtype is not None else None
    )
    groups = 8 if bias_dtype == torch.bfloat16 else 1
    ids = [torch.empty((tokens, k), dtype=torch.int16, device="cuda") for _ in range(2)]

    def call(expert, replay, count=tokens):
        return expert.apply(
            q[:count],
            w13,
            w2,
            logits[:count],
            MoEActivation.SITU,
            e,
            None,
            sf[:count],
            False,
            num_expert_group=groups,
            topk_group=min(groups, 2),
            e_score_correction_bias=bias,
            routed_scaling_factor=1.5,
            routing_replay_out=replay,
        )

    def materialize(output):
        assert isinstance(output, torch.Tensor) != defer
        if isinstance(output, torch.Tensor):
            return (output.clone(),)
        rows = output.expanded_idx_to_permuted_idx.long()
        return output.gemm2_permuted[rows].clone(), output.expert_weights.clone()

    prepared_calls = 0
    if tokens <= 8:
        module = get_trtllm_moe_sm100_module()
        tuner = AutoTuner.get()
        choose_one = tuner.choose_one
        forward = module.DABodyRunner.forward_from_metadata

        def choose_prepared_tactic(op, runners, tuning_config, inputs, **kwargs):
            if op == "flashinfer::trtllm_fp4_block_scale_moe":
                runner = runners[0]
                # Exercise the prepared ABI independently of timing-based tuning.
                tactic = next(t for t in runner.get_valid_tactics(inputs, None) if t[0] == 8)
                return runner, tactic
            return choose_one(op, runners, tuning_config, inputs, **kwargs)

        def run_prepared_body(self, *args, **kwargs):
            nonlocal prepared_calls
            prepared_calls += 1
            return forward(self, *args, **kwargs)

        monkeypatch.setattr(tuner, "choose_one", choose_prepared_tactic)
        monkeypatch.setattr(module.DABodyRunner, "forward_from_metadata", run_prepared_body)

    call(ordinary, ids[0])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call(compressed, ids[1])
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        before_capture = prepared_calls
        with torch.cuda.graph(graph, stream=stream):
            captured = call(compressed, ids[1])
        if tokens <= 8:
            assert prepared_calls == before_capture + 1
    workspace.current_workspace_manager().lock()
    for scale in (1.0, 0.0, 1e-5):
        logits.normal_(generator=generator).mul_(scale)
        if bias is not None:
            bias.normal_(generator=generator).mul_(scale * 0.1)
        expected = materialize(call(ordinary, ids[0]))
        graph.replay()
        actual = materialize(captured)
        assert torch.equal(ids[0], ids[1])
        for reference, result in zip(expected, actual):
            assert torch.equal(reference, result)

    if tokens == 8:
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for count in (3, 5):
                expected = materialize(call(ordinary, ids[0][:count], count))
                actual = materialize(call(compressed, ids[1][:count], count))
                assert torch.equal(ids[0][:count], ids[1][:count])
                for reference, result in zip(expected, actual):
                    assert torch.equal(reference, result)
