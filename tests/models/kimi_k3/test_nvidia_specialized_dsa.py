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
