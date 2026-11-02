# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from aphrodite.platforms import current_platform
from tests.models.utils import EmbedModelInfo
from tests.quantization.utils import is_quant_method_supported

from .mteb_embed_utils import mteb_test_embed_models


def _fp8_scaled_mm_unsupported() -> bool:
    """Whether the ModernBERT online FP8 test should be skipped.

    ``is_quant_method_supported("fp8")`` returns True on MI250 (gfx90a), but
    serving a per-tensor FP8 linear layer on ROCm requires CDNA3+ or RDNA4,
    so treat it as unsupported here.
    """
    if not is_quant_method_supported("fp8"):
        return True
    if current_platform.is_rocm():
        from aphrodite.platforms.rocm import get_cdna_version, on_gfx12x

        return get_cdna_version() <= 2 and not on_gfx12x()
    return False


MODEL_INFO = EmbedModelInfo(
    "Alibaba-NLP/gte-modernbert-base",
    mteb_score=0.748193353,
    architecture="ModernBertModel",
    seq_pooling_type="CLS",
    attn_type="encoder_only",
    is_prefix_caching_supported=False,
    is_chunked_prefill_supported=False,
    enable_test=True,
)


def _assert_modernbert_online_fp8(model) -> None:
    from aphrodite.model_executor.layers.quantization.online.fp8 import (
        Fp8PerTensorOnlineLinearMethod,
    )

    layer = model.encoder_layer.layers[0]
    linears = {
        "attn.Wqkv": layer.attn.Wqkv,
        "attn.Wo": layer.attn.Wo,
        "mlp.Wi": layer.mlp.Wi,
        "mlp.Wo": layer.mlp.Wo,
    }
    for name, linear in linears.items():
        assert isinstance(linear.quant_method, Fp8PerTensorOnlineLinearMethod), (
            f"{name} was not quantized with online FP8"
        )
        if current_platform.is_xpu():
            from aphrodite.model_executor.kernels.linear.scaled_mm.xpu import (
                XPUW8A8FP8LinearKernel,
            )

            assert isinstance(linear.quant_method.fp8_linear, XPUW8A8FP8LinearKernel)


@pytest.mark.flaky(reruns=2)
@pytest.mark.skipif(
    _fp8_scaled_mm_unsupported(),
    reason="No FP8 ScaledMM kernel is available on this GPU type.",
)
def test_modernbert_online_fp8_mteb(hf_runner, aphrodite_runner, monkeypatch) -> None:
    monkeypatch.setenv("APHRODITE_ALLOW_INSECURE_SERIALIZATION", "1")

    def check_online_fp8(aphrodite_model) -> None:
        aphrodite_model.apply_model(_assert_modernbert_online_fp8)

    aphrodite_extra_kwargs: dict[str, object] = {"quantization": "fp8_per_tensor"}
    if current_platform.is_xpu():
        aphrodite_extra_kwargs["kernel_config"] = {"linear_backend": "xpu"}

    mteb_test_embed_models(
        hf_runner,
        aphrodite_runner,
        MODEL_INFO,
        aphrodite_extra_kwargs=aphrodite_extra_kwargs,
        # Account for Online FP8 kernel variance across GPU architectures.
        atol=2e-3,
        aphrodite_model_callback=check_online_fp8,
    )
