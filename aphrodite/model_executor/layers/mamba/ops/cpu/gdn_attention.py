# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch
import torch.nn.functional as F

import aphrodite._custom_ops as ops
from aphrodite.forward_context import ForwardContext, get_forward_context
from aphrodite.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from aphrodite.model_executor.layers.mamba.ops.cpu.causal_conv1d import (
    causal_conv1d_fn_cpu as causal_conv1d_torch,
)
from aphrodite.model_executor.layers.mamba.ops.cpu.causal_conv1d import (
    causal_conv1d_update_cpu,
    causal_conv1d_update_torch,
)
from aphrodite.platforms import CpuArchEnum, current_platform
from aphrodite.utils.torch_utils import (
    LayerNameType,
    _resolve_layer_name,
    direct_register_custom_op,
)
from aphrodite.v1.attention.backends.gdn_attn import GDNAttentionMetadata

_CPU_GDN_ATTENTION_OPS_REGISTERED = False


def cpu_gdn_attention_core(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """CPU custom op for the core GDN attention computation."""
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    layer = forward_context.no_compile_layers[layer_name]

    attn_metadata = forward_context.attn_metadata

    if attn_metadata is None:
        return

    assert isinstance(attn_metadata, dict)
    attn_metadata_i = attn_metadata[layer.prefix]
    assert isinstance(attn_metadata_i, GDNAttentionMetadata)

    if attn_metadata_i.num_actual_tokens == 0:
        return

    assert mixed_qkv.dtype == torch.bfloat16, "CPU GDN attention requires BF16."

    conv_weight = getattr(layer.conv1d, "_cpu_unpacked_conv_weight", layer.conv1d.weight)
    width = conv_weight.size(-1)
    conv_cache = layer.kv_cache[0]
    if is_conv_state_dim_first():
        state_len = conv_cache.size(-1)
    else:
        state_len = conv_cache.size(-2)

    spec_decode_cache = state_len > (width - 1)
    if not spec_decode_cache:
        _cpu_gdn_attention_nonspec(layer, attn_metadata_i, mixed_qkv, b, a, core_attn_out)
        return

    _cpu_gdn_attention_spec_aware(
        layer,
        attn_metadata_i,
        mixed_qkv,
        b,
        a,
        core_attn_out,
        width,
        state_len,
    )


def _cpu_gdn_attention_nonspec(
    layer,
    attn_metadata_i: GDNAttentionMetadata,
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
) -> None:
    assert attn_metadata_i.spec_sequence_masks is None and attn_metadata_i.num_accepted_tokens is None, (
        "speculative decode not supported without a wide conv-state cache."
    )

    state_indices_tensor = attn_metadata_i.non_spec_state_indices_tensor
    query_start_loc = attn_metadata_i.non_spec_query_start_loc
    assert state_indices_tensor is not None
    assert query_start_loc is not None

    is_amx = torch.cpu._is_amx_tile_supported()

    conv_state = layer.kv_cache[0]
    if is_amx:
        # AMX causal conv requires [num_allocated_slots, kernel - 1, conv_dim].
        if is_conv_state_dim_first():
            raise RuntimeError("AMX GDN attention requires `SD` conv_state layout.")
        conv_state = conv_state.transpose(1, 2)
    else:
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)
        conv_weights = layer.conv1d.weight.view(layer.conv1d.weight.size(0), layer.conv1d.weight.size(2))

    # [num_allocated_slots, num_v_heads / tp_size, v_dim, k_dim]
    ssm_state = layer.kv_cache[1]
    mixed_qkv = mixed_qkv.contiguous()
    a = a.contiguous()
    b = b.contiguous()

    num_allocated_slots, head_num, v_dim, k_dim = ssm_state.size()
    ssm_state = ssm_state.view(
        num_allocated_slots,
        head_num,
        k_dim,
        v_dim,
    )

    num_decodes = attn_metadata_i.num_decodes
    num_decode_tokens = attn_metadata_i.num_decode_tokens
    num_prefills = attn_metadata_i.num_prefills
    num_prefill_tokens = attn_metadata_i.num_prefill_tokens

    # all decode requests (batched)
    if num_decodes > 0:
        decode_mixed_qkv = mixed_qkv[:num_decode_tokens]
        decode_b = b[:num_decode_tokens]
        decode_a = a[:num_decode_tokens]
        decode_state_indices = state_indices_tensor[:num_decodes]
        if is_amx:
            decode_mixed_qkv = ops.causal_conv1d_update_cpu(
                x=decode_mixed_qkv,
                conv_states=conv_state,
                weight=layer.conv1d.weight,
                bias=layer.conv1d.bias,
                silu_activation=layer.activation == "silu",
                conv_state_indices=decode_state_indices,
                is_vnni=True,
            )
        else:
            if current_platform.get_cpu_architecture() == CpuArchEnum.ARM:
                decode_conv_state = conv_state[decode_state_indices].contiguous()
                decode_mixed_qkv = causal_conv1d_update_torch(
                    x=decode_mixed_qkv.unsqueeze(-1),
                    conv_state=decode_conv_state,
                    weight=conv_weights,
                    bias=layer.conv1d.bias,
                    activation=layer.activation,
                ).squeeze(-1)
                conv_state[decode_state_indices] = decode_conv_state
            else:
                decode_mixed_qkv = causal_conv1d_update_cpu(
                    x=decode_mixed_qkv,
                    conv_state=conv_state,
                    weight=conv_weights,
                    bias=layer.conv1d.bias,
                    activation=layer.activation,
                    conv_state_indices=decode_state_indices,
                )

        query, key, value = layer.rearrange_mixed_qkv(decode_mixed_qkv)

        attn_out = ops.fused_sigmoid_gating_delta_rule_update_cpu(
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            q=query,
            k=key,
            v=value,
            a=decode_a,
            b=decode_b,
            initial_state_source=ssm_state,
            initial_state_indices=decode_state_indices,
            cu_seqlens=query_start_loc[: num_decodes + 1],
            use_qk_l2norm_in_kernel=True,
        )
        core_attn_out[:num_decode_tokens] = attn_out.squeeze(1)

    # all prefill requests: (varlen) currently naively loops over sequences
    if num_prefills > 0:
        has_initial_state = attn_metadata_i.has_initial_state
        assert has_initial_state is not None

        prefill_token_start = num_decode_tokens
        prefill_token_end = prefill_token_start + num_prefill_tokens
        prefill_mixed_qkv = mixed_qkv[prefill_token_start:prefill_token_end]
        prefill_b = b[prefill_token_start:prefill_token_end]
        prefill_a = a[prefill_token_start:prefill_token_end]
        prefill_state_indices = state_indices_tensor[num_decodes : num_decodes + num_prefills]
        prefill_query_start_loc = query_start_loc[num_decodes : num_decodes + num_prefills + 1] - num_decode_tokens
        prefill_has_initial_state = has_initial_state[num_decodes : num_decodes + num_prefills]

        if is_amx:
            prefill_mixed_qkv = ops.causal_conv1d_fwd_cpu(
                x=prefill_mixed_qkv.transpose(0, 1),
                weight=layer.conv1d.weight,
                bias=layer.conv1d.bias,
                conv_states=conv_state,
                query_start_loc=prefill_query_start_loc,
                cache_indices=prefill_state_indices,
                has_initial_state=prefill_has_initial_state,
                silu_activation=layer.activation == "silu",
                is_vnni=True,
            ).transpose(0, 1)
        else:
            prefill_mixed_qkv = causal_conv1d_torch(
                x=prefill_mixed_qkv.transpose(0, 1),
                weight=conv_weights,
                bias=layer.conv1d.bias,
                conv_states=conv_state,
                query_start_loc=prefill_query_start_loc,
                cache_indices=prefill_state_indices,
                has_initial_state=prefill_has_initial_state,
                activation=layer.activation,
            ).transpose(0, 1)

        query, key, value = layer.rearrange_mixed_qkv(prefill_mixed_qkv)
        g, beta = ops.fused_gdn_gating_cpu(A_log=layer.A_log, a=prefill_a, b=prefill_b, dt_bias=layer.dt_bias)

        initial_state = ssm_state[prefill_state_indices]
        initial_state[~prefill_has_initial_state, ...] = 0
        attn_out, last_recurrent_state = ops.chunk_gated_delta_rule_cpu(
            query=query,
            key=key,
            value=value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=prefill_query_start_loc,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )
        ssm_state[prefill_state_indices] = last_recurrent_state.to(ssm_state.dtype, copy=False)
        core_attn_out[prefill_token_start:prefill_token_end] = attn_out.squeeze(0)


def _conv_buffer_view(layer) -> torch.Tensor:
    """Return the conv-state cache as (num_slots, dim, state_len)."""
    conv_cache = layer.kv_cache[0]
    if is_conv_state_dim_first():
        return conv_cache
    return conv_cache.transpose(-1, -2)


def _ssm_state_view(layer) -> torch.Tensor:
    ssm_state = layer.kv_cache[1]
    num_slots, head_num, v_dim, k_dim = ssm_state.size()
    return ssm_state.view(num_slots, head_num, k_dim, v_dim)


def _unpacked_conv_weight(layer) -> torch.Tensor:
    w = getattr(layer.conv1d, "_cpu_unpacked_conv_weight", None)
    if w is not None:
        return w
    w = layer.conv1d.weight
    if w.dim() == 3:
        return w.view(w.size(0), w.size(2))
    return w


def _cpu_gdn_attention_spec_aware(
    layer,
    attn_metadata_i: GDNAttentionMetadata,
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    width: int,
    state_len: int,
) -> None:
    mixed_qkv = mixed_qkv.contiguous()
    a = a.contiguous()
    b = b.contiguous()

    spec_sequence_masks = attn_metadata_i.spec_sequence_masks
    conv_buf = _conv_buffer_view(layer)
    ssm_state = _ssm_state_view(layer)

    if spec_sequence_masks is None:
        _spec_aware_nonspec(
            layer,
            attn_metadata_i,
            mixed_qkv,
            b,
            a,
            core_attn_out,
            conv_buf,
            ssm_state,
            width,
        )
        return

    spec_token_indx = attn_metadata_i.spec_token_indx
    non_spec_token_indx = attn_metadata_i.non_spec_token_indx
    num_prefills = attn_metadata_i.num_prefills
    num_decodes = attn_metadata_i.num_decodes

    if num_prefills == 0 and num_decodes == 0:
        mixed_qkv_spec = mixed_qkv
        b_spec = b
        a_spec = a
        spec_out_indx = None
    else:
        assert spec_token_indx is not None
        mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
        b_spec = b.index_select(0, spec_token_indx)
        a_spec = a.index_select(0, spec_token_indx)
        spec_out_indx = spec_token_indx

    spec_out = _spec_forward(
        layer,
        attn_metadata_i,
        mixed_qkv_spec,
        b_spec,
        a_spec,
        conv_buf,
        ssm_state,
        width,
        state_len,
    )

    nonspec_out = None
    if (num_prefills > 0 or num_decodes > 0) and non_spec_token_indx is not None:
        mixed_qkv_ns = mixed_qkv.index_select(0, non_spec_token_indx)
        b_ns = b.index_select(0, non_spec_token_indx)
        a_ns = a.index_select(0, non_spec_token_indx)
        nonspec_out = _spec_aware_nonspec_subset(
            layer,
            attn_metadata_i,
            mixed_qkv_ns,
            b_ns,
            a_ns,
            conv_buf,
            ssm_state,
            width,
        )

    if spec_out_indx is None:
        core_attn_out[: spec_out.size(0)] = spec_out
    else:
        core_attn_out.index_copy_(0, spec_out_indx, spec_out)
        if nonspec_out is not None:
            assert non_spec_token_indx is not None
            core_attn_out.index_copy_(0, non_spec_token_indx, nonspec_out)


def _spec_forward(
    layer,
    attn_metadata_i: GDNAttentionMetadata,
    mixed_qkv_spec: torch.Tensor,
    b_spec: torch.Tensor,
    a_spec: torch.Tensor,
    conv_buf: torch.Tensor,
    ssm_state: torch.Tensor,
    width: int,
    state_len: int,
) -> torch.Tensor:
    num_spec_decodes = attn_metadata_i.num_spec_decodes
    spec_state_indices = attn_metadata_i.spec_state_indices_tensor
    spec_qsl = attn_metadata_i.spec_query_start_loc
    num_accepted = attn_metadata_i.num_accepted_tokens
    assert spec_state_indices is not None
    assert spec_qsl is not None
    assert num_accepted is not None

    spec_qsl_cpu = spec_qsl[: num_spec_decodes + 1].to("cpu", torch.int64)
    num_acc_cpu = num_accepted[:num_spec_decodes].to("cpu", torch.int64)
    seq_starts = spec_qsl_cpu[:-1]
    seq_lens = spec_qsl_cpu[1:] - spec_qsl_cpu[:-1]

    w2d = _unpacked_conv_weight(layer)
    dim = w2d.size(0)
    w = w2d.unsqueeze(1)
    bias = layer.conv1d.bias
    silu = layer.activation == "silu"

    conv_out = torch.empty_like(mixed_qkv_spec)
    col0 = spec_state_indices[:, 0].to("cpu", torch.int64)
    for i in range(num_spec_decodes):
        q_i = int(seq_lens[i].item())
        if q_i == 0:
            continue
        start = int(seq_starts[i].item())
        slot0 = int(col0[i].item())
        a_prev = int(num_acc_cpu[i].item())
        offset = a_prev - 1
        conv_state = conv_buf[slot0]
        x_seq = mixed_qkv_spec[start : start + q_i].transpose(0, 1).to(conv_state.dtype)
        prior = conv_state[:, offset : offset + (width - 1)]
        conv_in = torch.cat([prior, x_seq], dim=-1).unsqueeze(0)
        out = F.conv1d(conv_in, w, bias, groups=dim)[0]
        if silu:
            out = F.silu(out)
        conv_out[start : start + q_i] = out.transpose(0, 1).to(conv_out.dtype)
        keep = conv_state[:, offset + 1 : offset + 1 + (state_len - q_i)]
        conv_state.copy_(torch.cat([keep, x_seq], dim=-1))

    query, key, value = layer.rearrange_mixed_qkv(conv_out)
    query = query.squeeze(0).contiguous()
    key = key.squeeze(0).contiguous()
    value = value.squeeze(0).contiguous()
    spec_idx = spec_state_indices[:num_spec_decodes].to(torch.int32).contiguous()
    num_acc = num_accepted[:num_spec_decodes].to(torch.int32).contiguous()
    cu = spec_qsl[: num_spec_decodes + 1].to(torch.int32).contiguous()
    return ops.fused_sigmoid_gating_delta_rule_update_spec_cpu(
        A_log=layer.A_log,
        dt_bias=layer.dt_bias,
        q=query,
        k=key,
        v=value,
        a=a_spec.contiguous(),
        b=b_spec.contiguous(),
        initial_state_source=ssm_state,
        spec_state_indices=spec_idx,
        num_accepted_tokens=num_acc,
        cu_seqlens=cu,
        use_qk_l2norm_in_kernel=True,
    )


def _core_attn_out_like(layer, mixed_qkv: torch.Tensor) -> torch.Tensor:
    return torch.zeros(
        (mixed_qkv.size(0), layer.num_v_heads // layer.tp_size, layer.head_v_dim),
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
    )


def _spec_aware_nonspec(
    layer,
    attn_metadata_i: GDNAttentionMetadata,
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    conv_buf: torch.Tensor,
    ssm_state: torch.Tensor,
    width: int,
) -> None:
    state_indices_tensor = attn_metadata_i.non_spec_state_indices_tensor
    query_start_loc = attn_metadata_i.non_spec_query_start_loc
    assert state_indices_tensor is not None
    assert query_start_loc is not None

    conv_weights = _unpacked_conv_weight(layer)

    num_decodes = attn_metadata_i.num_decodes
    num_decode_tokens = attn_metadata_i.num_decode_tokens
    num_prefills = attn_metadata_i.num_prefills
    num_prefill_tokens = attn_metadata_i.num_prefill_tokens

    if num_decodes > 0:
        decode_mixed_qkv = mixed_qkv[:num_decode_tokens]
        decode_b = b[:num_decode_tokens]
        decode_a = a[:num_decode_tokens]
        decode_state_indices = state_indices_tensor[:num_decodes]
        if current_platform.get_cpu_architecture() == CpuArchEnum.ARM:
            conv_state_view = conv_buf[:, :, : width - 1]
            decode_conv_state = conv_state_view[decode_state_indices].contiguous()
            decode_mixed_qkv = causal_conv1d_update_torch(
                x=decode_mixed_qkv.unsqueeze(-1),
                conv_state=decode_conv_state,
                weight=conv_weights,
                bias=layer.conv1d.bias,
                activation=layer.activation,
            ).squeeze(-1)
            conv_state_view[decode_state_indices] = decode_conv_state
        else:
            decode_mixed_qkv = causal_conv1d_update_cpu(
                x=decode_mixed_qkv,
                conv_state=conv_buf[:, :, : width - 1],
                weight=conv_weights,
                bias=layer.conv1d.bias,
                activation=layer.activation,
                conv_state_indices=decode_state_indices,
            )

        query, key, value = layer.rearrange_mixed_qkv(decode_mixed_qkv)
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        attn_out = ops.fused_sigmoid_gating_delta_rule_update_cpu(
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            q=query,
            k=key,
            v=value,
            a=decode_a.contiguous(),
            b=decode_b.contiguous(),
            initial_state_source=ssm_state,
            initial_state_indices=decode_state_indices,
            cu_seqlens=query_start_loc[: num_decodes + 1],
            use_qk_l2norm_in_kernel=True,
        )
        core_attn_out[:num_decode_tokens] = attn_out.squeeze(1)

    if num_prefills > 0:
        has_initial_state = attn_metadata_i.has_initial_state
        assert has_initial_state is not None
        prefill_token_start = num_decode_tokens
        prefill_token_end = prefill_token_start + num_prefill_tokens
        prefill_mixed_qkv = mixed_qkv[prefill_token_start:prefill_token_end]
        prefill_b = b[prefill_token_start:prefill_token_end]
        prefill_a = a[prefill_token_start:prefill_token_end]
        prefill_state_indices = state_indices_tensor[num_decodes : num_decodes + num_prefills]
        prefill_query_start_loc = query_start_loc[num_decodes : num_decodes + num_prefills + 1] - num_decode_tokens
        prefill_has_initial_state = has_initial_state[num_decodes : num_decodes + num_prefills]
        prefill_mixed_qkv = causal_conv1d_torch(
            x=prefill_mixed_qkv.transpose(0, 1),
            weight=conv_weights,
            bias=layer.conv1d.bias,
            conv_states=conv_buf,
            query_start_loc=prefill_query_start_loc,
            cache_indices=prefill_state_indices,
            has_initial_state=prefill_has_initial_state,
            activation=layer.activation,
        ).transpose(0, 1)

        query, key, value = layer.rearrange_mixed_qkv(prefill_mixed_qkv)
        g, beta = ops.fused_gdn_gating_cpu(A_log=layer.A_log, a=prefill_a, b=prefill_b, dt_bias=layer.dt_bias)
        initial_state = ssm_state[prefill_state_indices]
        initial_state[~prefill_has_initial_state, ...] = 0
        attn_out, last_recurrent_state = ops.chunk_gated_delta_rule_cpu(
            query=query,
            key=key,
            value=value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=prefill_query_start_loc,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )
        ssm_state[prefill_state_indices] = last_recurrent_state.to(ssm_state.dtype, copy=False)
        core_attn_out[prefill_token_start:prefill_token_end] = attn_out.squeeze(0)


def _spec_aware_nonspec_subset(
    layer,
    attn_metadata_i: GDNAttentionMetadata,
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    conv_buf: torch.Tensor,
    ssm_state: torch.Tensor,
    width: int,
) -> torch.Tensor:
    del width
    out = _core_attn_out_like(layer, mixed_qkv)
    prefill_state_indices = attn_metadata_i.prefill_state_indices
    prefill_qsl = attn_metadata_i.prefill_query_start_loc
    prefill_has_initial_state = attn_metadata_i.prefill_has_initial_state
    assert prefill_state_indices is not None
    assert prefill_qsl is not None
    assert prefill_has_initial_state is not None

    conv_weights = _unpacked_conv_weight(layer)
    conv_out = causal_conv1d_torch(
        x=mixed_qkv.transpose(0, 1),
        weight=conv_weights,
        bias=layer.conv1d.bias,
        conv_states=conv_buf,
        query_start_loc=prefill_qsl,
        cache_indices=prefill_state_indices,
        has_initial_state=prefill_has_initial_state,
        activation=layer.activation,
    ).transpose(0, 1)

    query, key, value = layer.rearrange_mixed_qkv(conv_out)
    g, beta = ops.fused_gdn_gating_cpu(A_log=layer.A_log, a=a, b=b, dt_bias=layer.dt_bias)
    initial_state = ssm_state[prefill_state_indices]
    initial_state[~prefill_has_initial_state, ...] = 0
    attn_out, last_recurrent_state = ops.chunk_gated_delta_rule_cpu(
        query=query,
        key=key,
        value=value,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=prefill_qsl,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )
    ssm_state[prefill_state_indices] = last_recurrent_state.to(ssm_state.dtype, copy=False)
    out[:] = attn_out.squeeze(0)
    return out


def cpu_gdn_attention_core_fake(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """Fake implementation for torch.compile."""
    return


def register_cpu_gdn_attention_ops() -> None:
    global _CPU_GDN_ATTENTION_OPS_REGISTERED
    if _CPU_GDN_ATTENTION_OPS_REGISTERED:
        return

    direct_register_custom_op(
        op_name="cpu_gdn_attention_core",
        op_func=cpu_gdn_attention_core,
        mutates_args=["core_attn_out"],
        fake_impl=cpu_gdn_attention_core_fake,
    )
    _CPU_GDN_ATTENTION_OPS_REGISTERED = True
