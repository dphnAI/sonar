# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch


def test_dsa_metadata_uses_inner_attention_name():
    pytest.importorskip("cutlass")
    from sonar.models.kimi_k3.nvidia.specialized_decode.runtime import decode

    md = SimpleNamespace(num_prefills=0, num_decodes=2, num_decode_tokens=12)
    attention = SimpleNamespace(
        dsa_indexer=object(),
        mla_attn=SimpleNamespace(mla_attn=SimpleNamespace(layer_name="layers.3.self_attn")),
    )
    context = SimpleNamespace(attn_metadata={"layers.3.self_attn": md})
    assert decode.decode_metadata(attention, context, 6, 12) is md
    assert decode.decode_metadata(attention, context, 6, 18) is None
    md.num_prefills = 1
    assert decode.decode_metadata(attention, context, 6, 12) is None


@pytest.mark.parametrize("dsa", [False, True])
@pytest.mark.parametrize("quantized", [False, True])
def test_native_provider_selection_preserves_prepared_bf16_dsa(dsa, quantized):
    pytest.importorskip("cutlass")
    from sonar.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    from sonar.model_executor.layers.quantization.online.nf12 import Nf12OnlineLinearMethod
    from sonar.models.kimi_k3.nvidia.specialized_decode.runtime.decode import SpecializedDecode

    projection = object.__new__(LinearBase)
    torch.nn.Module.__init__(projection)
    method = Nf12OnlineLinearMethod if quantized else UnquantizedLinearMethod
    projection.quant_method = object.__new__(method)
    attention = torch.nn.Module()
    if dsa:
        attention.dsa_indexer = torch.nn.Module()
    mlp = torch.nn.Sequential(projection)
    layers = (SimpleNamespace(self_attn=attention, mlp=mlp),)
    assert SpecializedDecode.needs_native_providers(layers) is (dsa and quantized)


def test_native_providers_skip_prepared_cache_layout_and_weight_copies():
    pytest.importorskip("cutlass")
    from sonar.models.kimi_k3.nvidia.specialized_decode.runtime.decode import SpecializedDecode

    runtime = object.__new__(SpecializedDecode)
    runtime.native_providers = True
    runtime.layers = (SimpleNamespace(layer_idx=0),)
    runtime.attention = {0: object()}
    runtime.cache_config = SimpleNamespace(num_gpu_blocks=None)
    runtime.prepare_kv()
    assert not runtime.kv_prepared
    runtime.cache_config.num_gpu_blocks = 16
    runtime.prepare_kv()
    assert runtime.kv_prepared
    runtime.process_weights_after_loading()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="Specialized AttnRes requires SM100/103",
)
@pytest.mark.parametrize("block_write", [False, True])
def test_native_provider_matches_post_attention_prefix_transition(block_write):
    pytest.importorskip("cutlass")
    from sonar.models.kimi_k3.nvidia.model import KimiDecoderLayer
    from sonar.models.kimi_k3.nvidia.specialized_decode.runtime.decode import SpecializedDecode

    generator = torch.Generator(device="cuda").manual_seed(27)

    def randn(*shape):
        return torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)

    norm = SimpleNamespace(weight=randn(7168), variance_epsilon=1e-6)
    layer = SimpleNamespace(
        layer_idx=0,
        self_attn=SimpleNamespace(dsa_indexer=object()),
        mlp=None,
        use_attn_res=True,
        mlp_res_norm=norm,
        mlp_res_proj=SimpleNamespace(weight=randn(1, 7168) * 0.01),
        post_attention_layernorm=SimpleNamespace(weight=randn(7168), variance_epsilon=1e-6),
        prev_valid_blocks=2,
        is_block_write_layer=block_write,
    )
    runtime = object.__new__(SpecializedDecode)
    runtime.backbone = SimpleNamespace(config=None)
    runtime.layers = (layer,)
    runtime.native_providers = True
    runtime.build_providers()
    delta, prefix = randn(6, 7168), randn(6, 7168)
    blocks = randn(6, 4, 7168)
    reference, expected_prefix, _ = KimiDecoderLayer._post_attn_norm(
        layer, delta.clone(), blocks.clone(), prefix.clone()
    )
    output = torch.empty_like(delta)
    runtime.after[0].run(delta, prefix, blocks, output)
    torch.testing.assert_close(prefix, expected_prefix, rtol=0, atol=0)
    torch.testing.assert_close(output, reference, rtol=0.02, atol=0.02)


@pytest.mark.parametrize("dep", [False, True])
def test_wide_provider_reload_refreshes_only_prepared_kda(monkeypatch, dep):
    pytest.importorskip("cutlass")
    from sonar.models.kimi_k3.nvidia.specialized_decode.runtime import decode

    refreshed = []

    class Kda:
        def process_weights_after_loading(self, attention):
            refreshed.append(attention)

    monkeypatch.setattr(decode, "PreparedKda", Kda)
    attention = object()
    runtime = object.__new__(decode.SpecializedDecode)
    runtime.native_providers = False
    runtime.dep = dep
    runtime.hybrid = None if dep else object()
    runtime.layers = (
        SimpleNamespace(layer_idx=0, self_attn=attention),
        SimpleNamespace(layer_idx=1),
    )
    runtime.attention = {0: SimpleNamespace(core=Kda())}
    runtime.process_weights_after_loading()
    assert refreshed == [attention]


@pytest.mark.parametrize(
    "tp,ep,conv_dtype,recover,replay,reason",
    [
        (1, False, torch.bfloat16, False, False, "TP8/16"),
        (8, True, torch.bfloat16, False, False, "expert parallelism"),
        (8, False, torch.float32, False, False, "BF16 conv"),
        (8, False, torch.bfloat16, True, False, "RecoverSSM"),
        (8, False, torch.bfloat16, False, True, "ReplaySSM"),
    ],
)
def test_native_providers_retain_deployment_guards(monkeypatch, tp, ep, conv_dtype, recover, replay, reason):
    pytest.importorskip("cutlass")
    from sonar.models.kimi_k3.nvidia.specialized_decode import runtime
    from sonar.models.kimi_k3.nvidia.specialized_decode.runtime import decode

    config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=tp, enable_expert_parallel=ep),
        cache_config=SimpleNamespace(use_replayssm=replay, use_kda_recoverssm=recover),
    )
    monkeypatch.setattr(decode, "get_current_sonar_config", lambda: config)
    monkeypatch.setattr(runtime, "kda_dtypes", lambda _: (conv_dtype, torch.float32))
    monkeypatch.setattr(decode.SpecializedDecode, "needs_native_providers", staticmethod(lambda _: True))
    backbone = SimpleNamespace(layers=[], start_layer=0, end_layer=0)
    with pytest.raises(ValueError, match=reason):
        decode.SpecializedDecode(backbone, None)
