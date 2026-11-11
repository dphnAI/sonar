# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from functools import partial
from typing import ClassVar

import torch
import torch.distributed as dist

from sonar.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
from sonar.distributed import get_tp_group
from sonar.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from sonar.model_executor.warmup.cutedsl_warmup import (
    CuTeDSLCompileUnit,
    register_cutedsl_warmup_provider,
)
from sonar.utils.torch_utils import aux_stream

_MAX_NUM_TOKENS = 128
_SKINNY_MAX_NUM_TOKENS = 5
_MMA_TILER_MN = (64, 32)
_GEMM_CLUSTER_MN = (1, 8)
_B_PRIME_STAGES = 2
_COLLECTIVE_TOKEN_CTAS = 32
_LAMPORT_COPY_CTAS = 32
_LAMPORT_COPY_THREADS = 128
_SUPPORTED_TP_SIZES = (8, 16)


@dataclass(frozen=True)
class KimiK3LatentMoETailContract:
    tp_group_id: int
    tp_size: int
    device: torch.device
    dtype: torch.dtype
    hidden_size: int
    latent_size: int
    max_num_tokens: int
    rms_eps: float
    experts_per_token: int
    nf12_weights: bool = False


class KimiK3LatentMoETailOp:
    """Process-wide cached K3 latent-MoE tail implementation."""

    _instances: ClassVar[dict[KimiK3LatentMoETailContract, "KimiK3LatentMoETailOp"]] = {}

    @classmethod
    def _contract_and_group(
        cls,
        *,
        hidden_size: int,
        latent_size: int,
        dtype: torch.dtype,
        device: torch.device,
        rms_eps: float,
        experts_per_token: int = 0,
        nf12_weights: bool = False,
    ) -> tuple[KimiK3LatentMoETailContract, dist.ProcessGroup]:
        tp = get_tp_group()
        group = tp.device_group
        device = torch.device(device)
        tp_device = torch.device(tp.device)
        if device != tp_device:
            raise ValueError(f"Input device {device} does not match TP device {tp_device}.")
        return (
            KimiK3LatentMoETailContract(
                tp_group_id=id(group),
                tp_size=dist.get_world_size(group),
                device=device,
                dtype=dtype,
                hidden_size=hidden_size,
                latent_size=latent_size,
                max_num_tokens=_MAX_NUM_TOKENS,
                rms_eps=float(rms_eps),
                experts_per_token=experts_per_token,
                nf12_weights=nf12_weights,
            ),
            group,
        )

    @classmethod
    def initialize(
        cls,
        *,
        hidden_size: int,
        latent_size: int,
        dtype: torch.dtype,
        device: torch.device,
        rms_eps: float,
        experts_per_token: int = 0,
        nf12_weights: bool = False,
    ) -> "KimiK3LatentMoETailOp":
        contract, group = cls._contract_and_group(
            hidden_size=hidden_size,
            latent_size=latent_size,
            experts_per_token=experts_per_token,
            dtype=dtype,
            device=device,
            rms_eps=rms_eps,
            nf12_weights=nf12_weights,
        )
        op = cls._instances.get(contract)
        if op is None:
            op = cls(contract, group)
            cls._instances[contract] = op
        return op

    def __init__(
        self,
        contract: KimiK3LatentMoETailContract,
        group: dist.ProcessGroup,
    ) -> None:
        if contract.tp_size not in _SUPPORTED_TP_SIZES:
            raise ValueError(f"K3 latent-MoE tail fusion requires TP in {_SUPPORTED_TP_SIZES}, got {contract.tp_size}.")
        if contract.device.type != "cuda":
            raise ValueError("K3 latent-MoE tail fusion requires a CUDA device.")
        if torch.cuda.get_device_capability(contract.device)[0] != 10:
            raise ValueError("K3 latent-MoE tail fusion requires SM100.")
        if contract.dtype != torch.bfloat16:
            raise ValueError("K3 latent-MoE tail fusion requires bfloat16.")
        if (contract.hidden_size, contract.latent_size) != (7168, 3584):
            raise ValueError("K3 latent-MoE tail fusion requires hidden_size=7168 and latent_size=3584.")

        self.contract = contract
        self.rank = dist.get_rank(group)
        self._overlap_nf12_decode = (
            torch.cuda.get_device_capability(contract.device) == (10, 3) and contract.tp_size == 8
        )
        self._decode_events = (torch.cuda.Event(), torch.cuda.Event())
        from .cute_dsl.latent_moe_tail import (
            AdaptiveUpProjectionKernel,
            CollectiveKernel,
            LamportCopyKernel,
        )

        with torch.accelerator.device_index(contract.device.index):
            self._collective = CollectiveKernel(
                group=group,
                rank=self.rank,
                tp_size=contract.tp_size,
                latent_dim=contract.latent_size,
                hidden_dim=contract.hidden_size,
                max_m=contract.max_num_tokens,
                max_token_ctas=_COLLECTIVE_TOKEN_CTAS,
                rms_eps=contract.rms_eps,
                fp32_internal=False,
                top_k=contract.experts_per_token,
            )
            self._up_projection = AdaptiveUpProjectionKernel(
                group=group,
                rank=self.rank,
                tp_size=contract.tp_size,
                latent_dim=contract.latent_size,
                hidden_dim=contract.hidden_size,
                max_m=contract.max_num_tokens,
                skinny_max_m=_SKINNY_MAX_NUM_TOKENS,
                mma_tiler_mn=_MMA_TILER_MN,
                cluster_shape_mn=_GEMM_CLUSTER_MN,
                b_prime_stages=_B_PRIME_STAGES,
            )
            self._lamport_copy = LamportCopyKernel(
                hidden_dim=contract.hidden_size,
                max_m=contract.max_num_tokens,
                ctas=_LAMPORT_COPY_CTAS,
                threads=_LAMPORT_COPY_THREADS,
            )
        register_cutedsl_warmup_provider(self)

    def get_cutedsl_warmup_compile_units(self) -> tuple[CuTeDSLCompileUnit, ...]:
        contract = self.contract
        skinny_units = tuple(
            CuTeDSLCompileUnit(
                name="K3 latent MoE tail Skinny up-projection",
                key=(
                    "k3-latent-moe-tail-skinny-up-projection",
                    contract,
                    m,
                    nf12,
                ),
                compile=partial(self._up_projection.compile_skinny, m, nf12=nf12),
            )
            for m in range(1, self._up_projection.skinny_max_m + 1)
            for nf12 in ((False, True) if contract.nf12_weights else (False,))
            if not nf12 or self._up_projection.supports_nf12(m)
        )
        return skinny_units + (
            CuTeDSLCompileUnit(
                name="K3 latent MoE tail dynamic up-projection",
                key=(
                    "k3-latent-moe-tail-dynamic-up-projection",
                    contract,
                    _MMA_TILER_MN,
                    _GEMM_CLUSTER_MN,
                    _B_PRIME_STAGES,
                ),
                compile=self._up_projection.compile_dynamic,
            ),
        )

    def __call__(
        self,
        routed_output: torch.Tensor | UnfinalizedMoEOutput,
        shared_output: torch.Tensor,
        rms_weight: torch.Tensor,
        local_up_weight: torch.Tensor,
        *,
        weight_escapes: torch.Tensor | None = None,
        weight_exp_base: int | None = None,
    ) -> torch.Tensor:
        self._validate_inputs(
            routed_output,
            shared_output,
            rms_weight,
            local_up_weight,
            weight_escapes,
            weight_exp_base,
        )
        num_tokens = (
            routed_output.expanded_idx_to_permuted_idx.shape[0]
            if isinstance(routed_output, UnfinalizedMoEOutput)
            else routed_output.shape[0]
        )
        nf12 = local_up_weight.dtype == torch.uint8
        decode_stream = None
        if nf12 and not self._up_projection.supports_nf12(num_tokens):
            # Decode the rank-local weight while the collective prepares its input.
            if self._overlap_nf12_decode and not BreakableCUDAGraphCapture.is_active():
                decode_stream = aux_stream()
            if decode_stream is None:
                local_up_weight = torch.ops.nf12.decode(local_up_weight, weight_escapes, weight_exp_base)
            else:
                self._decode_events[0].record()
                with torch.cuda.stream(decode_stream):
                    self._decode_events[0].wait()
                    local_up_weight = torch.ops.nf12.decode(local_up_weight, weight_escapes, weight_exp_base)
                    self._decode_events[1].record()
            weight_escapes = weight_exp_base = None
            nf12 = False
        self._up_projection.ensure_compiled(num_tokens, nf12=nf12)
        latent, shared_shard = self._collective(
            routed_output,
            shared_output,
            rms_weight,
        )
        if decode_stream is not None:
            self._decode_events[1].wait()
        mailbox = self._up_projection(
            latent,
            local_up_weight,
            shared_shard,
            weight_escapes=weight_escapes,
            weight_exp_base=weight_exp_base,
        )
        if decode_stream is not None:
            # Return the temporary allocation to its stream after the last read.
            # This permits reuse across layers during graph capture.
            self._decode_events[0].record()
            self._decode_events[0].wait(decode_stream)
        return self._lamport_copy(
            mailbox,
            m=num_tokens,
        ).squeeze(0)

    def _validate_inputs(
        self,
        routed_output: torch.Tensor | UnfinalizedMoEOutput,
        shared_output: torch.Tensor,
        rms_weight: torch.Tensor,
        local_up_weight: torch.Tensor,
        weight_escapes: torch.Tensor | None,
        weight_exp_base: int | None,
    ) -> None:
        contract = self.contract
        if isinstance(routed_output, UnfinalizedMoEOutput):
            num_tokens = routed_output.expanded_idx_to_permuted_idx.shape[0]
        else:
            if routed_output.ndim != 2:
                raise ValueError("routed_output must be a 2D tensor.")
            num_tokens = routed_output.shape[0]
            if routed_output.shape != (num_tokens, contract.latent_size):
                raise ValueError(f"routed_output must have shape [M, {contract.latent_size}].")
        if shared_output.shape != (num_tokens, contract.hidden_size):
            raise ValueError(f"shared_output must have shape [M, {contract.hidden_size}].")
        if rms_weight.shape != (contract.latent_size,):
            raise ValueError(f"rms_weight must have shape [{contract.latent_size}].")
        local_hidden_size = contract.hidden_size // contract.tp_size
        nf12 = local_up_weight.dtype == torch.uint8
        weight_columns = contract.latent_size * 3 // 2 if nf12 else contract.latent_size
        if local_up_weight.shape != (local_hidden_size, weight_columns):
            raise ValueError(f"local_up_weight must have shape [{local_hidden_size}, {weight_columns}].")
        if nf12:
            if weight_escapes is None or weight_exp_base is None:
                raise ValueError("NF12 weights require the original escape table and exponent base.")
            if (
                weight_escapes.dtype != torch.int16
                or weight_escapes.ndim != 2
                or weight_escapes.shape[1] != 8
                or not isinstance(weight_exp_base, int)
                or not 0 <= weight_exp_base <= 120
                or weight_escapes.device != contract.device
                or not weight_escapes.is_contiguous()
            ):
                raise ValueError("NF12 metadata requires contiguous CUDA int16 [E, 8] and an integer base in [0, 120].")
        elif weight_escapes is not None or weight_exp_base is not None:
            raise ValueError("BF16 weights must not supply NF12 metadata.")
        if not 1 <= num_tokens <= contract.max_num_tokens:
            raise ValueError(f"K3 latent-MoE tail fusion requires between 1 and {contract.max_num_tokens} tokens.")

        tensors = (shared_output, rms_weight, local_up_weight)
        if any(tensor.device != contract.device for tensor in tensors):
            raise ValueError("All inputs must be on the contract device.")
        if (
            shared_output.dtype != contract.dtype
            or rms_weight.dtype != contract.dtype
            or local_up_weight.dtype != (torch.uint8 if nf12 else contract.dtype)
        ):
            raise ValueError("Inputs must use the contract dtype, except packed NF12 weights.")
        if any(not tensor.is_contiguous() for tensor in tensors):
            raise ValueError("All inputs must be contiguous.")
