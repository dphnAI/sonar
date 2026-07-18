# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
from collections.abc import Callable
from typing import ParamSpec

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import PatternMatcherPass

import aphrodite.ir.ops
from aphrodite.config import AphroditeConfig, get_layers_from_aphrodite_config
from aphrodite.config.utils import Range
from aphrodite.logger import init_logger
from aphrodite.model_executor.layers.attention.attention import (
    Attention,
    get_attention_context,
)
from aphrodite.platforms import current_platform
from aphrodite.utils.torch_utils import (
    _USE_LAYERNAME,
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)

from ..aphrodite_inductor_pass import AphroditeInductorPass, AphroditePatternMatcherPass
from ..inductor_pass import enable_fake_mode
from .matcher_utils import MatcherRotaryEmbedding
from .rms_quant_fusion import empty_bf16, empty_fp32, empty_i64

logger = init_logger(__name__)

P = ParamSpec("P")

SUPPORTED_FUSED_QK_NORM_ROPE_KVCACHE_HEAD_DIMS: tuple[int, ...] = (64, 128, 256)


def fused_qk_norm_rope_and_unified_kv_cache_update_impl(
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    layer_name: LayerNameType,
) -> torch.Tensor:
    layer_name = _resolve_layer_name(layer_name)
    _, attn_layer, kv_cache, layer_slot_mapping = get_attention_context(layer_name)
    if layer_slot_mapping is not None:
        attn_layer.impl.do_qk_norm_rope_kvcache_update(
            attn_layer,
            qkv,
            q_out,
            k_out,
            positions,
            q_weight,
            k_weight,
            rms_norm_eps,
            cos_sin_cache,
            is_neox,
            kv_cache,
            layer_slot_mapping,
        )
    else:
        q_out.zero_()
        k_out.zero_()

    return torch.empty(0, device=qkv.device, dtype=qkv.dtype)


def fused_qk_norm_rope_and_unified_kv_cache_update_fake(
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    layer_name: LayerNameType,
) -> torch.Tensor:
    return torch.empty(0, device=qkv.device, dtype=qkv.dtype)


direct_register_custom_op(
    op_name="fused_qk_norm_rope_and_unified_kv_cache_update",
    op_func=fused_qk_norm_rope_and_unified_kv_cache_update_impl,
    mutates_args=["q_out", "k_out"],
    fake_impl=fused_qk_norm_rope_and_unified_kv_cache_update_fake,
)


class QkNormRopeKvCachePattern:
    """
    Match Q/K RMSNorm + RoPE + unified KV cache update and replace it with
    AITER's fused QK-norm/RoPE/cache kernel.
    """

    FUSED_OP = torch.ops.aphrodite.fused_qk_norm_rope_and_unified_kv_cache_update.default

    def __init__(
        self,
        layer: Attention,
        eps: float,
        is_neox: bool,
        quant_query: bool,
    ) -> None:
        self.layer_name = layer.layer_name
        self.num_heads = layer.num_heads
        self.num_kv_heads = layer.num_kv_heads
        self.head_size = layer.head_size
        self.head_size_v = layer.head_size_v
        self.eps = eps
        self.is_neox = is_neox
        self.quant_query = quant_query

        self.q_size = self.num_heads * self.head_size
        self.k_size = self.num_kv_heads * self.head_size
        self.v_size = self.num_kv_heads * self.head_size_v

        self.rope_matcher = MatcherRotaryEmbedding(
            is_neox=is_neox,
            head_size=self.head_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
        )

    def get_inputs(self) -> list[torch.Tensor]:
        qkv = empty_bf16(5, self.q_size + self.k_size + self.v_size)
        positions = empty_i64(5)
        q_weight = empty_bf16(1, self.head_size)
        k_weight = empty_bf16(1, self.head_size)
        cos_sin_cache = empty_bf16(4096, self.head_size)
        inputs = [qkv, positions, q_weight, k_weight, cos_sin_cache]
        if self.quant_query:
            inputs.append(empty_fp32(1))
        if _USE_LAYERNAME:
            inputs.append(_encode_layer_name(self.layer_name))
        return inputs

    def pattern_non_fp8_quant_query(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        layer_name: LayerNameType,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)
        q_by_head = q.view(-1, self.num_heads, self.head_size)
        q_normed = aphrodite.ir.ops.rms_norm(q_by_head, q_weight, self.eps)
        q_flat = q_normed.view(-1, self.q_size)

        k_by_head = k.view(-1, self.num_kv_heads, self.head_size)
        k_normed = aphrodite.ir.ops.rms_norm(k_by_head, k_weight, self.eps)
        k_flat = k_normed.view(-1, self.k_size)

        q_rope, k_rope = self.rope_matcher(positions, q_flat, k_flat, cos_sin_cache)

        q_rope = q_rope.view(-1, self.num_heads, self.head_size)
        k_rope = k_rope.view(-1, self.num_kv_heads, self.head_size)
        v = v.view(-1, self.num_kv_heads, self.head_size_v)
        dummy = torch.ops.aphrodite.unified_kv_cache_update(k_rope, v, layer_name)
        return dummy, q_rope, k_rope, v

    def replacement_non_fp8_quant_query(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        layer_name: LayerNameType,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q_out = torch.empty(
            qkv.shape[0],
            self.num_heads,
            self.head_size,
            device=qkv.device,
            dtype=qkv.dtype,
        )
        k_out = torch.empty(
            qkv.shape[0],
            self.num_kv_heads,
            self.head_size,
            device=qkv.device,
            dtype=qkv.dtype,
        )
        _, _, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)
        v = v.view(qkv.shape[0], self.num_kv_heads, self.head_size_v)
        results = auto_functionalized(
            self.FUSED_OP,
            q_out=q_out,
            k_out=k_out,
            qkv=qkv,
            positions=positions,
            q_weight=q_weight,
            k_weight=k_weight,
            rms_norm_eps=self.eps,
            cos_sin_cache=cos_sin_cache,
            is_neox=self.is_neox,
            layer_name=layer_name,
        )
        return results[0], results[1], results[2], v

    def pattern_fp8_quant_query(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        q_scale: torch.Tensor,
        layer_name: LayerNameType,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)
        q_by_head = q.view(-1, self.num_heads, self.head_size)
        q_normed = aphrodite.ir.ops.rms_norm(q_by_head, q_weight, self.eps)
        q_flat = q_normed.view(-1, self.q_size)

        k_by_head = k.view(-1, self.num_kv_heads, self.head_size)
        k_normed = aphrodite.ir.ops.rms_norm(k_by_head, k_weight, self.eps)
        k_flat = k_normed.view(-1, self.k_size)

        q_rope, k_rope = self.rope_matcher(positions, q_flat, k_flat, cos_sin_cache)
        q_out = torch.empty_like(q_rope, dtype=current_platform.fp8_dtype())
        q_quant = auto_functionalized(
            torch.ops.aphrodite.rocm_aiter_per_tensor_quant.default,
            out=q_out,
            x=q_rope,
            scale=q_scale,
            is_dynamic=False,
        )
        q_rope_fp8 = q_quant[1]
        q_scale_out = q_quant[2]

        k_rope = k_rope.view(-1, self.num_kv_heads, self.head_size)
        v = v.view(-1, self.num_kv_heads, self.head_size_v)
        dummy = torch.ops.aphrodite.unified_kv_cache_update(k_rope, v, layer_name)
        return dummy, q_rope_fp8, k_rope, v, q_scale_out

    def replacement_fp8_quant_query(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        q_scale: torch.Tensor,
        layer_name: LayerNameType,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q_out = torch.empty(
            qkv.shape[0],
            self.num_heads,
            self.head_size,
            device=qkv.device,
            dtype=qkv.dtype,
        )
        k_out = torch.empty(
            qkv.shape[0],
            self.num_kv_heads,
            self.head_size,
            device=qkv.device,
            dtype=qkv.dtype,
        )
        _, _, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)
        v = v.view(qkv.shape[0], self.num_kv_heads, self.head_size_v)
        results = auto_functionalized(
            self.FUSED_OP,
            q_out=q_out,
            k_out=k_out,
            qkv=qkv,
            positions=positions,
            q_weight=q_weight,
            k_weight=k_weight,
            rms_norm_eps=self.eps,
            cos_sin_cache=cos_sin_cache,
            is_neox=self.is_neox,
            layer_name=layer_name,
        )
        q_fp8_flat = results[1].view(-1, self.q_size)
        q_fp8_out = torch.empty_like(q_fp8_flat, dtype=current_platform.fp8_dtype())
        q_requant = auto_functionalized(
            torch.ops.aphrodite.rocm_aiter_per_tensor_quant.default,
            out=q_fp8_out,
            x=q_fp8_flat,
            scale=q_scale,
            is_dynamic=False,
        )
        q_fp8 = q_requant[1]
        q_scale_out = q_requant[2]
        return results[0], q_fp8, results[2], v, q_scale_out

    @staticmethod
    def wrap_trace_fn(
        trace_fn: Callable[P, fx.GraphModule],
        *process_fx_fns: Callable[[fx.GraphModule], None],
    ) -> Callable[P, fx.GraphModule]:
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> fx.GraphModule:
            gm = trace_fn(*args, **kwargs)
            for process_fx in process_fx_fns:
                process_fx(gm)
            return gm

        return wrapped

    @staticmethod
    def fx_view_to_reshape(gm: torch.fx.GraphModule) -> None:
        from torch._inductor.fx_passes.post_grad import view_to_reshape

        view_to_reshape(gm)

    def _register(self, pattern, replacement, pm_pass) -> None:
        trace_fn = QkNormRopeKvCachePattern.wrap_trace_fn(
            pm.fwd_only,
            QkNormRopeKvCachePattern.fx_view_to_reshape,
        )
        inputs = self.get_inputs()
        argnames = [*inspect.signature(pattern).parameters.keys()]
        search_gm = trace_fn(pattern, inputs)
        search_fn_pattern = pm.fx_to_pattern(
            search_gm,
            ignore_types=(int, torch.SymInt),
            argnames=argnames,
        )
        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            trace_fn,
            pm_pass,
            search_fn_pattern=search_fn_pattern,
        )

    def register(self, pm_pass: PatternMatcherPass) -> None:
        _ln = _encode_layer_name(self.layer_name)

        if self.quant_query:
            if _USE_LAYERNAME:

                def pattern_q_with_layer_name(qkv, positions, q_weight, k_weight, cos_sin_cache, q_scale, layer_name):
                    return self.pattern_fp8_quant_query(
                        qkv, positions, q_weight, k_weight, cos_sin_cache, q_scale, layer_name
                    )

                def replacement_q_with_layer_name(
                    qkv, positions, q_weight, k_weight, cos_sin_cache, q_scale, layer_name
                ):
                    return self.replacement_fp8_quant_query(
                        qkv, positions, q_weight, k_weight, cos_sin_cache, q_scale, layer_name
                    )

                self._register(pattern_q_with_layer_name, replacement_q_with_layer_name, pm_pass)
            else:

                def pattern_q(qkv, positions, q_weight, k_weight, cos_sin_cache, q_scale):
                    return self.pattern_fp8_quant_query(qkv, positions, q_weight, k_weight, cos_sin_cache, q_scale, _ln)

                def replacement_q(qkv, positions, q_weight, k_weight, cos_sin_cache, q_scale):
                    return self.replacement_fp8_quant_query(
                        qkv, positions, q_weight, k_weight, cos_sin_cache, q_scale, _ln
                    )

                self._register(pattern_q, replacement_q, pm_pass)
        else:
            if _USE_LAYERNAME:

                def pattern_noq_with_layer_name(qkv, positions, q_weight, k_weight, cos_sin_cache, layer_name):
                    return self.pattern_non_fp8_quant_query(
                        qkv, positions, q_weight, k_weight, cos_sin_cache, layer_name
                    )

                def replacement_noq_with_layer_name(qkv, positions, q_weight, k_weight, cos_sin_cache, layer_name):
                    return self.replacement_non_fp8_quant_query(
                        qkv, positions, q_weight, k_weight, cos_sin_cache, layer_name
                    )

                self._register(pattern_noq_with_layer_name, replacement_noq_with_layer_name, pm_pass)
            else:

                def pattern_noq(qkv, positions, q_weight, k_weight, cos_sin_cache):
                    return self.pattern_non_fp8_quant_query(qkv, positions, q_weight, k_weight, cos_sin_cache, _ln)

                def replacement_noq(qkv, positions, q_weight, k_weight, cos_sin_cache):
                    return self.replacement_non_fp8_quant_query(qkv, positions, q_weight, k_weight, cos_sin_cache, _ln)

                self._register(pattern_noq, replacement_noq, pm_pass)


class QkNormRopeKvCacheFusionPass(AphroditePatternMatcherPass):
    """Fuse QK-norm + RoPE + KV cache update into a single AITER HIP kernel."""

    @enable_fake_mode
    def __init__(self, config: AphroditeConfig) -> None:
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(pass_name="qk_norm_rope_kvcache_fusion_pass")

        cc = config.compilation_config
        self.max_token_num = cc.pass_config.rope_kvcache_fusion_max_token_num

        dtype = config.model_config.dtype
        if dtype not in (torch.bfloat16, torch.float16):
            logger.warning_once("QK Norm+RoPE+KVCache fusion not enabled: unsupported dtype %s", dtype)
            return

        attn_layers = get_layers_from_aphrodite_config(config, Attention)
        for _, layer in attn_layers.items():
            if not layer.impl.fused_qk_norm_rope_kvcache_supported():
                continue
            if layer.head_size not in SUPPORTED_FUSED_QK_NORM_ROPE_KVCACHE_HEAD_DIMS:
                logger.warning_once(
                    "QK Norm+RoPE+KVCache fusion not enabled for a layer: "
                    "head_size=%d is not supported by the "
                    "fused_qk_norm_rope_cache_pts_quant_shuffle kernel "
                    "(supported: %s). Falling back to the unfused path.",
                    layer.head_size,
                    SUPPORTED_FUSED_QK_NORM_ROPE_KVCACHE_HEAD_DIMS,
                )
                continue
            if layer.head_size_v != layer.head_size:
                logger.warning_once(
                    "QK Norm+RoPE+KVCache fusion not enabled for a layer: "
                    "head_size_v=%d differs from head_size=%d, which the fused "
                    "kernel does not support. Falling back to the unfused path.",
                    layer.head_size_v,
                    layer.head_size,
                )
                continue
            for epsilon in [1e-5, 1e-6]:
                for neox in [True, False]:
                    for quant_q in [False, True]:
                        QkNormRopeKvCachePattern(
                            layer=layer,
                            eps=epsilon,
                            is_neox=neox,
                            quant_query=quant_q,
                        ).register(self.patterns)
            if _USE_LAYERNAME:
                break

        self.dump_patterns(config, self.patterns)

    @AphroditeInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.info(
            "QK-Norm+RoPE+KVCache fusion: replaced %s pattern(s) with AITER fused_qk_norm_rope_cache_pts_quant_shuffle",
            self.matched_count,
        )

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return compile_range.end <= self.max_token_num

    def uuid(self) -> str:
        return AphroditeInductorPass.hash_source(self, QkNormRopeKvCachePattern)
