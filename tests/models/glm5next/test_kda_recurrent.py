# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash KDA recurrent (decode) kernel.

The decode path hands the kernel column slices of the merged `q|k|v` conv
output and of the fused `qkvbfg_a` projection (beta), so q/k/v/beta are
token-strided rather than contiguous. Both backends must match a pure-PyTorch
recurrence, read these slices in place and reject unsupported layouts.
"""

import pytest
import torch

from sonar.platforms import current_platform

if current_platform.is_rocm():
    from sonar.models.glm5next.amd.ops.third_party.kda import fused_recurrent_kda
else:
    from sonar.models.glm5next.nvidia.ops.third_party.kda import fused_recurrent_kda

pytestmark = pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="Requires CUDA or ROCm")

H, D = 16, 128
LOWER_BOUND = -5.0


def naive_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    a_log: torch.Tensor,
    g_bias: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 reference for one sequence: `[T, H, D]` inputs, `[H, D, D]`
    (v-major) state; mirrors the kernel's in-kernel gate, beta sigmoid and
    q/k l2norm.
    """
    q, k, v, raw_g, raw_beta = (x.float() for x in (q, k, v, raw_g, raw_beta))
    q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6) * D**-0.5
    k = k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    gate = LOWER_BOUND * torch.sigmoid(a_log.exp()[:, None] * (raw_g + g_bias))
    beta = torch.sigmoid(raw_beta)
    s = state.clone()
    out = torch.empty_like(v)
    for t in range(q.shape[0]):
        s = s * gate[t].exp()[:, None, :]
        u = beta[t][:, None] * (v[t] - torch.einsum("hvk,hk->hv", s, k[t]))
        s = s + u[:, :, None] * k[t][:, None, :]
        out[t] = torch.einsum("hvk,hk->hv", s, q[t])
    return out, s


def make_inputs(num_seqs: int, query_len: int, device: torch.device):
    """Token-strided q/k/v/beta as the decode path produces them: column
    slices of a merged `[T, q|k|v]` conv output and of the fused
    `[T, qkv|beta|f_a|g_a]` projection.
    """
    T, proj = num_seqs * query_len, H * D
    qkv = torch.randn(T, 3 * proj, dtype=torch.bfloat16, device=device)
    projected = torch.randn(T, 3 * proj + H + 2 * D, dtype=torch.bfloat16, device=device)
    q, k, v = (x.view(1, T, H, D) for x in qkv.split(proj, dim=-1))
    beta = projected[:, 3 * proj : 3 * proj + H].unsqueeze(0)
    # (A size-1 token dim gets an arbitrary stride from `view`.)
    assert T == 1 or (q.stride(1) == 3 * proj and beta.stride(1) == projected.stride(0))
    inputs = dict(
        q=q,
        k=k,
        v=v,
        g=torch.randn(1, T, H, D, dtype=torch.bfloat16, device=device),
        beta=beta,
        a_log=0.5 * torch.randn(H, dtype=torch.float32, device=device),
        g_bias=0.1 * torch.randn(H * D, dtype=torch.float32, device=device),
        cu_seqlens=torch.arange(0, T + 1, query_len, dtype=torch.int32, device=device),
    )
    # Slot 0 is NULL_BLOCK_ID; sequences own random distinct slots (one per
    # token in the spec-decode layout).
    slots = torch.randperm(T, device=device).to(torch.int32) + 1
    if query_len == 1:
        inputs["ssm_state_indices"] = slots
    else:
        inputs["ssm_state_indices"] = slots.view(num_seqs, query_len)
        inputs["num_accepted_tokens"] = torch.randint(1, query_len + 1, (num_seqs,), dtype=torch.int32, device=device)
    state = torch.randn(T + 1, H, D, D, dtype=torch.float32, device=device)
    return inputs, state


def run_kernel(inputs: dict, state: torch.Tensor) -> torch.Tensor:
    out, _ = fused_recurrent_kda(
        **inputs,
        initial_state=state,
        use_qk_l2norm_in_kernel=True,
        sigmoid_beta=True,
        compute_gate=True,
        lower_bound=LOWER_BOUND,
    )
    return out


@pytest.mark.parametrize(("num_seqs", "query_len"), [(1, 1), (7, 1), (3, 3)], ids=["1x1", "7x1", "3x3"])
@torch.inference_mode()
def test_fused_recurrent_kda_matches_reference(num_seqs: int, query_len: int):
    torch.manual_seed(0)
    device = torch.device("cuda")
    inputs, state = make_inputs(num_seqs, query_len, device)
    expected_state = state.clone()
    out = run_kernel(inputs, state)

    indices = inputs["ssm_state_indices"].view(num_seqs, query_len)
    accepted = inputs.get("num_accepted_tokens")
    expected = torch.empty_like(out[0], dtype=torch.float32)
    for n in range(num_seqs):
        first = indices[n, 0 if accepted is None else accepted[n] - 1]
        s = expected_state[first]
        for t in range(query_len):
            tok = slice(n * query_len + t, n * query_len + t + 1)
            expected[tok], s = naive_recurrent_kda(
                inputs["q"][0, tok],
                inputs["k"][0, tok],
                inputs["v"][0, tok],
                inputs["g"][0, tok],
                inputs["beta"][0, tok],
                inputs["a_log"],
                inputs["g_bias"].view(H, D),
                s,
            )
            expected_state[indices[n, t]] = s

    torch.testing.assert_close(out[0].float(), expected, rtol=1e-2, atol=1e-3)
    used = indices.flatten().long()
    torch.testing.assert_close(state[used], expected_state[used], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(("num_seqs", "query_len"), [(7, 1), (3, 3)], ids=["7x1", "3x3"])
@torch.inference_mode()
def test_fused_recurrent_kda_strided_inputs_bit_identical_to_contiguous(num_seqs: int, query_len: int):
    torch.manual_seed(0)
    device = torch.device("cuda")
    inputs, state = make_inputs(num_seqs, query_len, device)
    for name in ("q", "k", "v", "beta"):
        assert not inputs[name].is_contiguous()
    contiguous = {name: x.contiguous() if name in ("q", "k", "v", "beta") else x for name, x in inputs.items()}
    state_ref = state.clone()
    out_ref = run_kernel(contiguous, state_ref)
    out = run_kernel(inputs, state)
    torch.testing.assert_close(out, out_ref, rtol=0, atol=0)
    torch.testing.assert_close(state, state_ref, rtol=0, atol=0)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="GLM compact state uses the NVIDIA RecoverSSM path")
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_glm_recoverssm_commits_only_accepted_updates(state_dtype):
    from types import SimpleNamespace

    from sonar.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from sonar.models.glm5next.common.kda import _recoverssm_view
    from sonar.models.kimi_k3.nvidia.ops.recoverssm import (
        KDARecoverSSMCommitContext,
        KDARecoverSSMRecords,
        kda_recoverssm_verify,
    )

    torch.manual_seed(5)
    num_seqs, window, history = 3, 3, 3
    device = torch.device("cuda")
    inputs, state = make_inputs(num_seqs, window, device)
    state = state.to(state_dtype)
    initial_state = state.clone()
    indices = inputs["ssm_state_indices"][:, 0].contiguous()
    accepted = torch.tensor([1, 2, 3], dtype=torch.int32, device=device)
    conv = torch.randn(state.shape[0], 3 * H * D, history + window - 1, device=device, dtype=torch.bfloat16)
    initial_conv = conv.clone()
    records = KDARecoverSSMRecords.allocate(num_seqs, H, window, D, torch.bfloat16, device)
    # GLM stores A_log broadcast-shaped, unlike Kimi's flat head vector.
    layer = SimpleNamespace(
        kv_cache=(conv if is_conv_state_dim_first() else conv.transpose(-1, -2), state),
        recoverssm_records=records,
        A_log=inputs["a_log"].view(1, 1, H, 1),
        dt_bias=inputs["g_bias"],
        local_num_heads=H,
        head_dim=D,
        kda_lower_bound=LOWER_BOUND,
    )
    view = _recoverssm_view(layer)
    output = torch.empty_like(inputs["v"])
    kda_recoverssm_verify(
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        raw_g=inputs["g"],
        raw_beta=inputs["beta"],
        A_log=view.A_log,
        dt_bias=view.dt_bias,
        lower_bound=view.gate_lower_bound,
        checkpoint_state=state,
        records=records,
        query_start_loc=inputs["cu_seqlens"],
        state_indices=indices,
        spec_query_len=window,
        out=output,
    )
    torch.testing.assert_close(state, initial_state, rtol=0, atol=0)
    context = KDARecoverSSMCommitContext.create([view], spec_query_len=window, max_num_reqs=num_seqs)
    context.commit(accepted, indices, inputs["cu_seqlens"])

    expected_state = initial_state.clone()
    expected_conv = initial_conv.clone()
    for row, count in enumerate(accepted.tolist()):
        slot = indices[row]
        tokens = slice(row * window, (row + 1) * window)
        expected_out, _ = naive_recurrent_kda(
            inputs["q"][0, tokens],
            inputs["k"][0, tokens],
            inputs["v"][0, tokens],
            inputs["g"][0, tokens],
            inputs["beta"][0, tokens],
            view.A_log,
            view.dt_bias.view(H, D),
            initial_state[slot].float(),
        )
        torch.testing.assert_close(output[0, tokens].float(), expected_out, rtol=3e-2, atol=3e-2)
        accepted_tokens = slice(row * window, row * window + count)
        _, committed = naive_recurrent_kda(
            inputs["q"][0, accepted_tokens],
            inputs["k"][0, accepted_tokens],
            inputs["v"][0, accepted_tokens],
            inputs["g"][0, accepted_tokens],
            inputs["beta"][0, accepted_tokens],
            view.A_log,
            view.dt_bias.view(H, D),
            initial_state[slot].float(),
        )
        expected_state[slot] = committed.to(state_dtype)
        expected_conv[slot, :, :history] = initial_conv[slot, :, count - 1 : count - 1 + history]
    torch.testing.assert_close(state.float(), expected_state.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(conv, expected_conv, rtol=0, atol=0)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="GLM compact state uses the NVIDIA RecoverSSM path")
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_glm_compact_verify_matches_per_position(monkeypatch, state_dtype):
    from types import SimpleNamespace

    from sonar.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from sonar.models.glm5next.common import kda
    from sonar.models.kimi_k3.nvidia.kda_metadata import KimiK3KDAMetadata
    from sonar.models.kimi_k3.nvidia.ops.recoverssm import (
        KDARecoverSSMCommitContext,
        KDARecoverSSMRecords,
    )

    torch.manual_seed(7)
    device = torch.device("cuda")
    num_seqs, window, history = 2, 3, 3
    inputs, initial_state = make_inputs(num_seqs, window, device)
    initial_state = initial_state.to(state_dtype)
    slots = inputs["ssm_state_indices"]
    qkv = torch.cat([inputs[name][0].flatten(1) for name in ("q", "k", "v")], dim=-1)
    initial_conv = torch.randn(
        initial_state.shape[0],
        3 * H * D,
        history + window - 1,
        dtype=torch.bfloat16,
        device=device,
    )
    weights = torch.randn(3 * H * D, 1, history + 1, device=device) * 0.1

    def run(compact):
        layer = object.__new__(kda.Glm5NextLinearAttention)
        torch.nn.Module.__init__(layer)
        layer.prefix = "layer.0"
        layer.local_num_heads, layer.head_dim = H, D
        layer.local_projection_size = H * D
        layer.num_spec = window - 1
        layer.kda_safe_gate = True
        layer.kda_lower_bound = LOWER_BOUND
        layer._conv_state_dim_first = is_conv_state_dim_first()
        layer.conv1d = SimpleNamespace(weight=weights, bias=None)
        layer.A_log = inputs["a_log"].view(1, 1, H, 1)
        layer.dt_bias = inputs["g_bias"]
        conv = initial_conv.clone()
        layer.kv_cache = (
            conv if layer._conv_state_dim_first else conv.transpose(-1, -2),
            initial_state.clone(),
        )
        layer.recoverssm_records = (
            KDARecoverSSMRecords.allocate(num_seqs, H, window, D, torch.bfloat16, device) if compact else None
        )
        metadata = KimiK3KDAMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decodes=0,
            num_decode_tokens=0,
            num_spec_decodes=num_seqs,
            num_spec_decode_tokens=num_seqs * window,
            num_actual_tokens=num_seqs * window,
            spec_query_start_loc=inputs["cu_seqlens"],
            spec_state_indices_tensor=slots[:, :1].contiguous() if compact else slots,
            spec_sequence_masks=None if compact else torch.ones(num_seqs, dtype=torch.bool, device=device),
            num_accepted_tokens=torch.ones(num_seqs, dtype=torch.int32, device=device),
        )
        monkeypatch.setattr(
            kda,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata={layer.prefix: metadata}),
        )
        out = torch.empty_like(inputs["v"])
        layer._forward(qkv.clone(), inputs["g"], inputs["beta"], out)
        return layer, out, conv

    reference, expected_output, expected_conv = run(False)
    compact, output, conv = run(True)
    torch.testing.assert_close(output.float(), expected_output.float(), rtol=3e-2, atol=3e-3)
    torch.testing.assert_close(conv, expected_conv, rtol=0, atol=0)
    torch.testing.assert_close(compact.kv_cache[1], initial_state, rtol=0, atol=0)

    accepted = torch.tensor([1, window], dtype=torch.int32, device=device)
    context = KDARecoverSSMCommitContext.create(
        [kda._recoverssm_view(compact)],
        spec_query_len=window,
        max_num_reqs=num_seqs,
    )
    context.commit(accepted, slots[:, 0].contiguous(), inputs["cu_seqlens"])
    committed_conv = expected_conv.clone()
    for row, count in enumerate(accepted.tolist()):
        first = slots[row, 0]
        torch.testing.assert_close(
            compact.kv_cache[1][first].float(),
            reference.kv_cache[1][slots[row, count - 1]].float(),
            rtol=2e-2,
            atol=2e-2,
        )
        committed_conv[first, :, :history] = expected_conv[first, :, count - 1 : count - 1 + history]
    torch.testing.assert_close(conv, committed_conv, rtol=0, atol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Both implementations require zero-copy token-strided layouts",
)
@torch.inference_mode()
def test_fused_recurrent_kda_rejects_unaddressable_layouts():
    """Layouts the token-stride addressing cannot express must fail loudly
    rather than read the wrong tokens: a batch slice of a wider buffer
    (`stride(0) != T * stride(1)`), overlapping tokens, and a head-strided
    (transposed) block.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    inputs, state = make_inputs(1, 1, device)
    T = 4
    base = torch.randn(4, T, 2 * H * D, dtype=torch.bfloat16, device=device)
    bad_q = {
        "batch slice": base[::2, :, : H * D].view(2, T, H, D),
        "overlapping tokens": base[:1, :, : H * D].as_strided((1, T, H, D), (0, D, D, 1)),
        "head-strided": base[:1, :, : H * D].view(1, T, D, H).transpose(2, 3),
    }
    for q in bad_q.values():
        broken = dict(inputs, q=q, k=q, v=q)
        broken["cu_seqlens"] = None if q.shape[0] > 1 else inputs["cu_seqlens"]
        with pytest.raises(AssertionError, match=r"torch.Size"):
            run_kernel(broken, state)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="The large-grid fix is NVIDIA-only")
@pytest.mark.parametrize("num_seqs,num_heads", [(1024, 64), (2048, 48)])
def test_fused_recurrent_kda_more_than_65535_programs(num_seqs: int, num_heads: int):
    """Regression test for the launch grid.

    GLM-5.3-Flash has 64 KDA value heads; at TP=1 (every DP rank) a decode
    batch of 1024 sequences needs 1024 * 64 = 65536 (sequence, head)
    programs. The kernel used to put that product into gridDim.z, whose CUDA
    limit is 65535, so the launch failed with `Triton Error [CUDA]: invalid
    argument` at CUDA-graph capture and DP deployments could not start. The
    grid now carries the product in gridDim.x. Small head dims keep the test
    cheap; the shape that matters is the number of programs. Every (sequence,
    head) pair is independent, so the full batch must be bitwise identical to
    the same batch run in two halves.
    """
    torch.manual_seed(0)
    dev = torch.device("cuda")
    head_dim = 16
    pool = num_seqs + 8
    q = torch.randn(1, num_seqs, num_heads, head_dim, device=dev, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = torch.randn_like(q)
    beta = torch.randn(1, num_seqs, num_heads, device=dev, dtype=torch.bfloat16)
    state = torch.randn(pool, num_heads, head_dim, head_dim, device=dev, dtype=torch.float32)
    cu = torch.arange(0, num_seqs + 1, device=dev, dtype=torch.int32)
    # slot 0 is NULL_BLOCK_ID and is skipped by the kernel: use slots 1..
    idx = (torch.randperm(pool - 1, device=dev)[:num_seqs] + 1).to(torch.int32)
    a_log = torch.randn(num_heads, device=dev, dtype=torch.float32)
    g_bias = torch.randn(num_heads, head_dim, device=dev, dtype=torch.float32)

    def run(sl: slice, cu_: torch.Tensor, st: torch.Tensor, out: torch.Tensor):
        fused_recurrent_kda(
            q=q[:, sl],
            k=k[:, sl],
            v=v[:, sl],
            g=g[:, sl],
            beta=beta[:, sl],
            initial_state=st,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_,
            ssm_state_indices=idx[sl],
            sigmoid_beta=True,
            a_log=a_log,
            g_bias=g_bias,
            compute_gate=True,
            lower_bound=-5.0,
            out=out,
        )

    full_state = state.clone()
    full_out = torch.zeros_like(k)
    run(slice(0, num_seqs), cu, full_state, full_out)

    half = num_seqs // 2
    ref_state = state.clone()
    ref_out = torch.zeros_like(k)
    for s, e in ((0, half), (half, num_seqs)):
        cu_h = torch.arange(0, e - s + 1, device=dev, dtype=torch.int32)
        part = torch.zeros_like(k[:, s:e])
        run(slice(s, e), cu_h, ref_state, part)
        ref_out[:, s:e] = part
    torch.accelerator.synchronize()
    assert torch.equal(full_out, ref_out)
    assert torch.equal(full_state, ref_state)


@pytest.mark.parametrize("offset", [16, 144, 256])
@pytest.mark.parametrize("dim_first", [False, True])
@pytest.mark.parametrize("num_spec", [0, 3])
@torch.inference_mode()
def test_prefill_checkpoint_resumes_suffix(monkeypatch, dim_first, num_spec, offset):
    """Restoring both cached states must reproduce an uninterrupted prefill."""
    from types import SimpleNamespace

    from sonar.model_executor.layers.mamba.checkpoint import (
        MambaPrefillCheckpointMetadata,
    )
    from sonar.model_executor.layers.mamba.kda_checkpoint import (
        FlashKDAPrefillCheckpointExporter,
    )
    from sonar.models.glm5next.common import kda
    from sonar.v1.attention.backends.gdn_attn import GDNAttentionMetadata
    from sonar.v1.attention.backends.utils import compute_causal_conv1d_metadata
    from sonar.v1.worker.workspace import WorkspaceManager

    if kda._resolve_kda_prefill_backend("auto", D, torch.bfloat16, -5) != "flashkda":
        pytest.skip("requires FlashKDA")
    import sonar._flashkda_C  # noqa: F401

    torch.manual_seed(42)
    device = torch.device("cuda")
    length, heads = 273, 2
    width, state_len = 3 * heads * D, 3 + num_spec
    layer = object.__new__(kda.Glm5NextLinearAttention)
    torch.nn.Module.__init__(layer)
    layer.prefix = "layer.0"
    layer.local_num_heads = heads
    layer.head_dim = D
    layer.conv_size = 4
    layer.local_projection_size = heads * D
    layer.kda_safe_gate = True
    layer.kda_lower_bound = -5.0
    layer.num_spec = num_spec
    layer.recoverssm_records = None
    layer.kda_prefill_backend = "flashkda"
    layer._conv_state_dim_first = dim_first
    layer.conv1d = SimpleNamespace(weight=torch.randn(width, 1, 4, device=device) * 0.1, bias=None)
    layer.A_log = torch.randn(1, 1, heads, 1, device=device) * 0.1
    layer.dt_bias = torch.randn(heads * D, device=device) * 0.1
    state_shape = (2, heads, D, D)
    layer._flashkda_buffer_specs = (
        (state_shape, torch.float32),
        (state_shape, torch.float32),
        (
            (torch.ops._flashkda_C.get_workspace_size(2 * length, heads, 2),),
            torch.uint8,
        ),
        ((1, 2 * length, heads, D), torch.bfloat16),
    )
    layer._checkpoint_exporter = FlashKDAPrefillCheckpointExporter(state_len=3)
    workspace = WorkspaceManager(device)
    monkeypatch.setattr(kda, "current_workspace_manager", lambda: workspace)
    conv_shape = (width, state_len) if dim_first else (state_len, width)
    num_slots = 5 + (num_spec + 1 if num_spec else 0)
    conv_storage = torch.zeros(num_slots, width * state_len + 8, device=device, dtype=torch.bfloat16)
    conv = conv_storage[:, : width * state_len].view(num_slots, *conv_shape)
    recurrent_storage = torch.zeros(num_slots, heads * D * D + 8, device=device, dtype=torch.float32)
    recurrent = recurrent_storage[:, : heads * D * D].view(num_slots, heads, D, D)
    layer.kv_cache = (conv, recurrent)
    projected = torch.randn(2 * length, width + heads, device=device, dtype=torch.bfloat16)
    raw_qkv, beta = projected[:, :width], projected[:, width:].unsqueeze(0)
    raw_before = raw_qkv.clone()
    gate = torch.randn(1, 2 * length, heads, D, device=device, dtype=torch.bfloat16)

    def run(qkv, g, b, slots, has_initial, checkpoint=None):
        n = qkv.shape[0] // 2
        cu_cpu = torch.tensor([0, n, 2 * n], dtype=torch.int32)
        nums, batch_ptr, chunk_offsets = compute_causal_conv1d_metadata(cu_cpu, device=device)
        metadata = GDNAttentionMetadata(
            num_prefills=2,
            num_prefill_tokens=2 * n,
            num_decodes=0,
            num_decode_tokens=0,
            num_spec_decodes=0,
            num_spec_decode_tokens=0,
            num_actual_tokens=2 * n,
            has_initial_state=torch.tensor(has_initial, device=device),
            non_spec_query_start_loc=cu_cpu.to(device),
            non_spec_state_indices_tensor=torch.tensor(slots, dtype=torch.int32, device=device),
            checkpoint=checkpoint,
            nums_dict=nums,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=chunk_offsets,
        )
        monkeypatch.setattr(
            kda,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata={layer.prefix: metadata}),
        )
        spec_tokens = num_spec + 1 if num_spec else 0
        if spec_tokens:
            qkv = torch.cat([qkv[:spec_tokens], qkv])
            g = torch.cat([g[:, :spec_tokens], g], dim=1)
            b = torch.cat([b[:, :spec_tokens], b], dim=1)
            metadata.num_spec_decodes = 1
            metadata.num_spec_decode_tokens = spec_tokens
            metadata.num_actual_tokens += spec_tokens
            metadata.spec_sequence_masks = torch.tensor([True, False, False], device=device)
            metadata.spec_query_start_loc = torch.tensor([0, spec_tokens], dtype=torch.int32, device=device)
            metadata.spec_state_indices_tensor = torch.arange(5, num_slots, dtype=torch.int32, device=device).unsqueeze(
                0
            )
            metadata.spec_token_indx = torch.arange(spec_tokens, device=device)
            metadata.non_spec_token_indx = torch.arange(spec_tokens, qkv.shape[0], device=device)
            metadata.num_accepted_tokens = torch.ones(1, dtype=torch.int32, device=device)
        out = torch.empty(1, qkv.shape[0], heads, D, device=device, dtype=torch.bfloat16)
        layer._forward(qkv, g, b, out)
        return out[:, spec_tokens:]

    checkpoint = MambaPrefillCheckpointMetadata(
        torch.tensor([offset, 0], dtype=torch.int32, device=device),
        torch.tensor([3, 0], dtype=torch.int32, device=device),
    )
    expected = run(raw_qkv, gate, beta, [1, 2], [False, False], checkpoint)
    expected_state = recurrent[1].clone()
    saved_conv, saved_recurrent = conv[3].clone(), recurrent[3].clone()
    torch.testing.assert_close(raw_qkv, raw_before)
    conv_view = conv if dim_first else conv.transpose(-1, -2)
    torch.testing.assert_close(conv_view[3, :, :3], raw_qkv[offset - 3 : offset].T)
    assert torch.count_nonzero(conv_view[3, :, 3:]) == 0
    assert torch.count_nonzero(conv[0]) == torch.count_nonzero(recurrent[0]) == 0

    run(
        torch.cat([raw_qkv[:offset], raw_qkv[length : length + offset]]),
        torch.cat([gate[:, :offset], gate[:, length : length + offset]], dim=1),
        torch.cat([beta[:, :offset], beta[:, length : length + offset]], dim=1),
        [1, 2],
        [False, False],
    )
    torch.testing.assert_close(saved_recurrent, recurrent[1], atol=0.003, rtol=0.03)

    suffix = slice(offset, length)
    qkv = torch.cat([raw_qkv[suffix], raw_qkv[length + offset :]])
    g = torch.cat([gate[:, suffix], gate[:, length + offset :]], dim=1)
    b = torch.cat([beta[:, suffix], beta[:, length + offset :]], dim=1)
    conv[4].copy_(saved_conv)
    recurrent[4].copy_(saved_recurrent)
    actual = run(qkv, g, b, [3, 4], [True, True])
    torch.testing.assert_close(actual[:, : length - offset], expected[:, suffix], atol=0.003, rtol=0.03)
    torch.testing.assert_close(recurrent[3], expected_state, atol=0.003, rtol=0.03)
