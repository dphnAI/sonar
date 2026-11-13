# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

from sonar.platforms import current_platform
from sonar.utils.network_utils import get_open_port


def _check_recurrent_pcp(rank: int, port: int, model: str) -> None:
    import sonar._flashkda_C  # noqa: F401

    from sonar.distributed import cleanup_dist_env_and_memory
    from sonar.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
    from sonar.model_executor.layers.mamba.checkpoint import MambaPrefillCheckpointMetadata
    from sonar.model_executor.layers.mamba.kda_checkpoint import FlashKDAPrefillCheckpointExporter
    from sonar.models.glm5next.common import kda as glm_kda
    from sonar.models.kimi_k3.nvidia import kda as kimi_kda
    from sonar.models.kimi_k3.nvidia.kda_metadata import KimiK3KDAMetadata
    from sonar.v1.attention.backends.gdn_attn import GDNAttentionMetadata
    from sonar.v1.attention.backends.utils import compute_causal_conv1d_metadata
    from sonar.v1.attention.ops.pcp import PCPTokenMapping
    from sonar.v1.worker.gpu.pcp_manager import PCPManager
    from sonar.v1.worker.workspace import WorkspaceManager
    from tests.utils import ensure_current_sonar_config

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    init_distributed_environment(2, rank, f"tcp://127.0.0.1:{port}", rank)
    try:
        with ensure_current_sonar_config():
            initialize_model_parallel(prefill_context_model_parallel_size=2)
        torch.manual_seed(42)
        is_kimi = model != "glm"
        module = kimi_kda if is_kimi else glm_kda
        layer_type = kimi_kda.KimiK3DeltaAttention if is_kimi else glm_kda.Glm5NextLinearAttention
        metadata_type = KimiK3KDAMetadata if is_kimi else GDNAttentionMetadata
        heads, dim = 12, 128
        width, channels = heads * dim, 3 * heads * dim
        layer = object.__new__(layer_type)
        torch.nn.Module.__init__(layer)
        layer.prefix = "layer.0"
        layer.local_num_heads, layer.head_dim = heads, dim
        layer.local_projection_size = width
        layer.conv_size, layer.num_spec = 4, 0
        layer.kda_safe_gate, layer.kda_lower_bound = True, -5.0
        layer.recoverssm_records = None
        layer.kda_prefill_backend = "flashkda"
        layer._conv_state_dim_first = True
        layer.conv1d = SimpleNamespace(weight=torch.randn(channels, 1, 4, device=device) * 0.1, bias=None)
        layer.A_log = torch.randn(1, 1, heads, 1, device=device) * 0.1
        layer.dt_bias = torch.randn(width, device=device) * 0.1
        state_shape = (2, heads, dim, dim)
        state_spec = (state_shape, torch.float32)
        workspace_spec = ((torch.ops._flashkda_C.get_workspace_size(512, heads, 2),), torch.uint8)
        output_spec = ((1, 512, heads, dim), torch.bfloat16)
        layer._flashkda_buffer_specs = (
            (output_spec, state_spec, state_spec, workspace_spec)
            if is_kimi
            else (state_spec, state_spec, workspace_spec, output_spec)
        )
        layer._checkpoint_exporter = FlashKDAPrefillCheckpointExporter(state_len=3)
        conv = torch.zeros(5, channels, 3, device=device, dtype=torch.bfloat16)
        state = torch.zeros(5, heads, dim, dim, device=device)
        layer.kv_cache = (conv, state)
        if is_kimi:
            layer.A_log = layer.A_log.reshape(heads)
            layer.gate_lower_bound = -5.0
            layer.use_recoverssm = False
            layer.kda_spec_decode_backend = "native"
            layer.kda_decode_backend = {
                "kimi": "triton",
                "kimi-native": "native",
                "kimi-flashinfer": "flashinfer",
            }[model]
            with ensure_current_sonar_config():
                layer.o_norm = kimi_kda.FusedRMSNormGated(dim, activation="sigmoid").to(device)
            layer.decode_conv1d_weight = None
            layer.decode_norm_weight = None
            if layer.kda_decode_backend != "triton":
                layer.decode_conv1d_weight = layer.conv1d.weight.view(3, width, 4).transpose(1, 2).contiguous()
                layer.decode_norm_weight = layer.o_norm.weight.detach().float().contiguous()
        manager = PCPManager(2, rank, device, shard_decode_requests=True)
        workspace = WorkspaceManager(device)
        checkpoint = MambaPrefillCheckpointMetadata(
            torch.tensor([144, 96], dtype=torch.int32, device=device),
            torch.tensor([3, 4], dtype=torch.int32, device=device),
        )
        sizes = (channels, width, width, heads) if is_kimi else (channels, width, heads)
        inputs = torch.randn(466, sum(sizes), device=device, dtype=torch.bfloat16)

        def run(projected: torch.Tensor, output: torch.Tensor) -> None:
            parts = projected.split(sizes, dim=-1)
            qkv, gate, beta = parts[0], parts[1].reshape(1, -1, heads, dim), parts[-1].unsqueeze(0)
            if is_kimi:
                layer._forward(qkv, gate, parts[2].reshape(-1, heads, dim), beta, output)
            else:
                layer._forward(qkv, gate, beta, output)

        saved_checkpoint = None
        with torch.inference_mode(), pytest.MonkeyPatch.context() as patch:
            patch.setattr(module, "current_workspace_manager", lambda: workspace)
            if is_kimi:
                patch.setattr(module, "is_conv_state_dim_first", lambda: True)
            for phase, lengths in (("prefill", [273, 193]), ("decode", [1, 1]), ("resume", [129, 97])):
                if phase == "prefill":
                    projected = inputs
                elif phase == "decode":
                    projected = inputs[:2]
                else:
                    assert saved_checkpoint is not None
                    conv[1:3].copy_(saved_checkpoint[0])
                    state[1:3].copy_(saved_checkpoint[1])
                    rows = torch.cat((torch.arange(144, 273), torch.arange(369, 466))).to(device)
                    projected = inputs[rows]
                initial_conv, initial_state = conv.clone(), state.clone()
                cu = torch.tensor([0, lengths[0], sum(lengths)], dtype=torch.int32)
                nums, batch_ptr, offsets = compute_causal_conv1d_metadata(cu, device=device)
                is_prefill = phase != "decode"
                metadata = metadata_type(
                    num_prefills=2 if is_prefill else 0,
                    num_prefill_tokens=sum(lengths) if is_prefill else 0,
                    num_decodes=0 if is_prefill else 2,
                    num_decode_tokens=0 if is_prefill else 2,
                    num_spec_decodes=0,
                    num_spec_decode_tokens=0,
                    num_actual_tokens=sum(lengths),
                    has_initial_state=torch.full((2,), phase != "prefill", device=device),
                    non_spec_query_start_loc=cu.to(device),
                    non_spec_state_indices_tensor=torch.tensor([1, 2], dtype=torch.int32, device=device),
                    checkpoint=checkpoint if phase == "prefill" else None,
                    nums_dict=nums,
                    batch_ptr=batch_ptr,
                    token_chunk_offset_ptr=offsets,
                )
                patch.setattr(
                    module,
                    "get_forward_context",
                    lambda metadata=metadata: SimpleNamespace(attn_metadata={layer.prefix: metadata}),
                )
                expected = torch.empty(1, sum(lengths), heads, dim, dtype=torch.bfloat16, device=device)
                # Preserve projection strides while isolating decode's in-place convolution.
                run(projected.clone(), expected)
                expected_conv, expected_state = conv.clone(), state.clone()
                _, counts = manager._build_batch_layout(
                    np.array(lengths, dtype=np.int32),
                    np.zeros(2, dtype=np.int32),
                    np.full(2, is_prefill, dtype=np.bool_),
                    cu.numpy(),
                )
                local_width = max(counts)
                assert manager._padded_gather_idx is not None
                assert manager._hidden_restore_idx is not None
                rows = manager._padded_gather_idx[rank * local_width : (rank + 1) * local_width]
                metadata.pcp_token_mapping = PCPTokenMapping(manager._hidden_restore_idx, rows)
                conv.copy_(initial_conv)
                state.copy_(initial_state)
                actual = torch.empty(1, local_width, heads, dim, dtype=torch.bfloat16, device=device)
                run(projected[rows], actual)
                torch.testing.assert_close(actual, expected[:, rows], rtol=0, atol=0)
                torch.testing.assert_close(conv, expected_conv, rtol=0, atol=0)
                torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
                if phase == "prefill":
                    saved_checkpoint = conv[3:5].clone(), state[3:5].clone()
    finally:
        cleanup_dist_env_and_memory()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA PCP and FlashKDA")
@pytest.mark.parametrize("model", ["glm", "kimi", "kimi-native", "kimi-flashinfer"])
def test_pcp_preserves_recurrent_checkpoints(model: str):
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two GPUs")
    from sonar.models.glm5next.common.kda import _resolve_kda_prefill_backend

    if _resolve_kda_prefill_backend("auto", 128, torch.bfloat16, -5) != "flashkda":
        pytest.skip("Requires FlashKDA")
    mp.spawn(_check_recurrent_pcp, args=(get_open_port(), model), nprocs=2)
