# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 specialized decode on MI355X (layer mode), enabled by
``additional_config={"decode_backend": "latency_specialized"}``.

``specialized_decode_config`` runs in the model's ``__init__``, before the weights
load. It returns None for the stock decode, and raises with every reason from
``unsupported`` when the specialized decode is requested for a deployment it cannot
serve. Checks that need the loaded weights run when the executor is built
(``decode.SpecializedDecode``), those that need the KV cache in ``prepare_kv``."""

from dataclasses import dataclass

import torch

from sonar.logger import init_logger

logger = init_logger(__name__)

AUTO, SPECIALIZED = "auto", "latency_specialized"
TP = 8
# a launch's x rows fit one CTA's LDS
MAX_TOKENS = 8
WHOLE, FFN = "whole", "ffn"
# Kimi-K3's text shapes the kernels are written for (hidden size, routed experts,
# KDA heads, KDA head dim, KDA short-conv width)
K3_SHAPES = (7168, 896, 96, 128, 4)


@dataclass(frozen=True)
class DecodeConfig:
    # rows a request verifies: num_speculative_tokens + 1
    q: int

    @property
    def widths(self) -> tuple[int, ...]:
        """The rows a launch serves: whole requests, at most MAX_TOKENS rows. Every
        uniform verify step up to MAX_TOKENS rows is one of them, so an eager step
        never pads; a FULL replay pads whole requests the way Sonar's stock decode
        does."""
        return tuple(range(self.q, MAX_TOKENS + 1, self.q))


def decode_backend(sonar_config) -> str:
    """The requested decode backend: the one place that reads the switch."""
    extra = sonar_config.additional_config
    return extra.get("decode_backend", AUTO) if isinstance(extra, dict) else AUTO


def specialized_decode_config(sonar_config) -> DecodeConfig | None:
    """The deployment's specialized decode settings, None for the stock decode.
    Raises when the specialized decode is requested for an unsupported deployment,
    listing every reason."""
    backend = decode_backend(sonar_config)
    if backend == AUTO:
        return None
    if backend != SPECIALIZED:
        raise ValueError(f"decode_backend must be '{AUTO}' or '{SPECIALIZED}', got {backend!r}")
    reasons = unsupported(sonar_config)
    if reasons:
        raise ValueError(
            "Kimi-K3 specialized decode (decode_backend=latency_specialized) does "
            "not support: " + "; ".join(reasons) + f" (decode_backend={AUTO} runs "
            "Sonar's decode)"
        )
    if piecewise_graphs_only(sonar_config):
        logger.warning(
            "Kimi-K3 specialized decode: PIECEWISE-only CUDA graphs capture every "
            "decode step as stock; use FULL decode graphs"
        )
    spec = sonar_config.speculative_config
    return DecodeConfig(q=spec.num_speculative_tokens + 1)


def unsupported(sonar_config) -> list[str]:
    """Every reason this deployment cannot run the specialized decode (empty: it
    can). Config only: the loaded weights are checked when the executor is built.
    Default-deny: every feature below is refused, the kernels serve none of them."""
    from sonar.config import CompilationMode
    from sonar.model_executor.layers.mamba.mamba_utils import (
        MambaStateDtypeCalculator,
    )

    mc, pc, cc = (
        sonar_config.model_config,
        sonar_config.parallel_config,
        sonar_config.cache_config,
    )
    spec, oc = sonar_config.speculative_config, sonar_config.offload_config
    kv = sonar_config.kv_transfer_config
    checks = (
        (pc.tensor_parallel_size != TP, f"TP {pc.tensor_parallel_size} (TP{TP} only)"),
        (pc.pipeline_parallel_size != 1, "pipeline parallelism"),
        (pc.data_parallel_size != 1, "data parallelism"),
        (pc.enable_expert_parallel, "expert parallelism"),
        (pc.enable_eplb, "EPLB"),
        (
            pc.decode_context_parallel_size not in (1, pc.tensor_parallel_size),
            "decode context parallelism other than 1 or TP",
        ),
        (pc.prefill_context_parallel_size != 1, "prefill context parallelism"),
        # the kernels read one batch's attention metadata per step
        (pc.use_ubatching, "DBO (dual-batch overlap)"),
        (sonar_config.lora_config is not None, "LoRA"),
        # MoRIIO consumers finish loads before execution, without per-layer hooks.
        (
            kv is not None
            and (
                kv.kv_connector != "MoRIIOConnector"
                or kv.kv_connector_module_path is not None
                or kv.kv_role != "kv_consumer"
            ),
            "KV transfer connectors other than a MoRIIO consumer",
        ),
        (
            sonar_config.aux_output_config.enable_return_routed_experts,
            "routed-expert capture",
        ),
        # the kernels read the loaded weights in place
        (mc.enable_sleep_mode, "sleep mode"),
        (
            oc.uva.cpu_offload_gb > 0 or oc.prefetch.offload_group_size > 0,
            "CPU weight offload",
        ),
        (sonar_config.weight_transfer_config is not None, "weight updates"),
        # the kernels serve spec-verify steps of whole uniform requests only
        (spec is None, "decoding without speculative decoding"),
        (
            spec is not None and spec.num_speculative_tokens + 1 > MAX_TOKENS,
            f"more than {MAX_TOKENS - 1} speculative tokens",
        ),
        (
            spec is not None and spec.enable_adaptive_verification,
            "adaptive verification",
        ),
        (
            spec is not None and spec.uses_dynamic_speculative_decoding(),
            "dynamic speculative decoding",
        ),
        # K1 keeps a BF16 conv state and an FP32 recurrent state in place; it
        # maintains no ReplaySSM / RecoverSSM records
        (
            MambaStateDtypeCalculator.kda_state_dtype(mc.dtype, cc.mamba_cache_dtype)
            != (torch.bfloat16, torch.float32),
            "KDA cache dtypes other than a BF16 conv state and an FP32 recurrent state (mamba cache dtype)",
        ),
        (cc.use_replayssm or cc.use_kda_recoverssm, "ReplaySSM / RecoverSSM"),
        # the model state hooks build the executor and bind it to each KV cache
        (
            not sonar_config.use_v2_model_runner,
            "the V1 model runner (set SONAR_USE_V2_MODEL_RUNNER=1)",
        ),
        # plan() and the layer kernels run in forward; a compiled forward would
        # trace them
        (
            sonar_config.compilation_config.mode == CompilationMode.STOCK_TORCH_COMPILE,
            "torch.compile (compilation mode STOCK_TORCH_COMPILE)",
        ),
    )
    reasons = [why for bad, why in checks if bad]
    shapes = text_shapes(mc.hf_text_config)
    if shapes != K3_SHAPES:
        reasons.append(
            f"model shapes {shapes} (hidden, experts, KDA heads, KDA head dim, conv width; Kimi-K3 is {K3_SHAPES})"
        )
    if getattr(mc.hf_text_config, "attn_res_block_size", None) is None:
        reasons.append("models without AttnRes")
    return reasons + platform_reasons()


def text_shapes(text) -> tuple:
    kda = getattr(text, "linear_attn_config", None) or {}
    return (
        getattr(text, "hidden_size", None),
        getattr(text, "num_experts", None),
        kda.get("num_heads"),
        kda.get("head_dim"),
        kda.get("short_conv_kernel_size"),
    )


def platform_reasons() -> list[str]:
    from sonar.platforms import current_platform

    if not current_platform.is_rocm():
        return ["this platform (ROCm only)"]
    from sonar.platforms.rocm import on_gfx950

    if not on_gfx950():
        return ["this GPU (gfx950 / MI355X only)"]
    from sonar.models.kimi_k3.amd.specialized_decode.mono.common.plan import BLOCKS

    cus = torch.cuda.get_device_properties().multi_processor_count
    if cus != BLOCKS:
        # every CTA of a launch must be resident: a partitioned GPU deadlocks
        return [f"{cus} CUs (a partitioned GPU; the kernels need {BLOCKS})"]
    return []


def piecewise_graphs_only(sonar_config) -> bool:
    """Decode steps captured as PIECEWISE graphs only: plan() runs them stock."""
    from sonar.config import CUDAGraphMode

    mode = sonar_config.compilation_config.cudagraph_mode
    if mode is None or mode == CUDAGraphMode.NONE:
        return False
    return mode.decode_mode() != CUDAGraphMode.FULL


def layer_kinds(config) -> dict[int, str]:
    """Decoder layer -> kind, for every layer a kernel serves: the MoE layers with
    AttnRes and at least one block before them, minus the MLA block-write layers
    (none in Kimi-K3)."""
    block = config.attn_res_block_size
    if block is None or config.num_experts is None:
        return {}
    out = {}
    for i in range(1, config.num_hidden_layers):
        if i < config.first_k_dense_replace or i % config.moe_layer_freq:
            continue
        if config.is_kda_layer(i):
            out[i] = WHOLE
        elif i % block:
            out[i] = FFN
    return out
