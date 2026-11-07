# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Callable, Set
from typing import TypeAlias

import torch
import torch.nn as nn

from sonar.config import get_current_sonar_config
from sonar.config.pooler import TokenPoolingType
from sonar.model_executor.layers.pooler import PoolingParamsUpdate
from sonar.tasks import PoolingTask
from sonar.utils.torch_utils import async_tensor_h2d
from sonar.v1.pool.metadata import PoolingMetadata

TokenPoolingMethodOutputItem: TypeAlias = torch.Tensor | None


class TokenPoolingMethod(nn.Module, ABC):
    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"token_embed", "token_classify"}

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate()

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> list[TokenPoolingMethodOutputItem]:
        raise NotImplementedError


class AllPool(TokenPoolingMethod):
    def __init__(self, output_sizes_fn: Callable[[], list[int]] | None = None):
        super().__init__()
        # Logical output widths in the current request order.
        self.output_sizes_fn = output_sizes_fn

        sonar_config = get_current_sonar_config()
        scheduler_config = sonar_config.scheduler_config

        self.enable_chunked_prefill = scheduler_config.enable_chunked_prefill
        # With async scheduling, the next step may overwrite the hidden states
        # buffer before outputs are copied to CPU, so they must own their storage.
        self.clone_finished = bool(scheduler_config.async_scheduling)

    def extra_repr(self) -> str:
        return f"enable_chunked_prefill={self.enable_chunked_prefill}, clone_finished={self.clone_finished}"

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> list[TokenPoolingMethodOutputItem]:
        pooling_cursor = pooling_metadata.get_pooling_cursor()
        # Use the already-CPU num_scheduled_tokens tensor so `.tolist()`
        # doesn't trigger a GPU->CPU sync.
        split_sizes = pooling_cursor.num_scheduled_tokens_cpu.tolist()
        if split_sizes:
            # DispatchPooler passes the full hidden_states tensor.
            # slice out the subgroup once, then split it by
            # per-request token counts.
            group_start = int(pooling_cursor.first_token_indices_gpu[0].item())
            group_end = int(pooling_cursor.last_token_indices_gpu[-1].item()) + 1
            hidden_states_group = hidden_states[group_start:group_end]
            hidden_states_lst = list(hidden_states_group.split(split_sizes))
        else:
            hidden_states_lst = []
        if self.output_sizes_fn is not None:
            hidden_states_lst = [
                hs[:, :size] for hs, size in zip(hidden_states_lst, self.output_sizes_fn(), strict=True)
            ]

        if not self.enable_chunked_prefill:
            if self.clone_finished:
                return [hs.clone() for hs in hidden_states_lst]
            return hidden_states_lst

        pooling_states = pooling_metadata.pooling_states
        finished_mask = pooling_cursor.get_finished_mask()

        # If chunked_prefill is enabled
        # 1. first store the chunked hidden_states in pooling_states.hidden_states_cache
        for p, hs_chunk, finished in zip(pooling_states, hidden_states_lst, finished_mask):
            # Own data retained across steps or returned without concatenation.
            needs_owned_storage = not finished or (self.clone_finished and not p.hidden_states_cache)
            p.hidden_states_cache.append(hs_chunk.clone() if needs_owned_storage else hs_chunk)

        # 2. Once prefill is finished, send hidden_states_cache to PoolerHead
        output_list = list[TokenPoolingMethodOutputItem]()
        for p, finished in zip(pooling_states, finished_mask):
            if finished:
                hidden_states_cache = p.hidden_states_cache
                if len(hidden_states_cache) == 1:
                    output_list.append(hidden_states_cache[0])
                else:
                    output_list.append(torch.concat(hidden_states_cache, dim=0))
                p.clean()
            else:
                output_list.append(None)

        return output_list


class StepPool(AllPool):
    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate(requires_token_ids=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> list[TokenPoolingMethodOutputItem]:
        pooled_data_lst = super().forward(hidden_states, pooling_metadata)
        # Use the CPU copy of prompt_token_ids so the step_tag_id mask can be
        # resolved to indices without a d2h sync from boolean indexing.
        prompt_token_ids_cpu = pooling_metadata.get_prompt_token_ids_cpu()
        pooling_params = pooling_metadata.pooling_params

        pooled_data = list[torch.Tensor | None]()
        for data, token_id_cpu, pooling_param in zip(pooled_data_lst, prompt_token_ids_cpu, pooling_params):
            # for unfinished chunked prefill
            if data is None:
                pooled_data.append(None)
            else:
                step_tag_id = pooling_param.step_tag_id
                returned_token_ids = pooling_param.returned_token_ids

                if returned_token_ids is not None and len(returned_token_ids) > 0:
                    data = data[:, returned_token_ids]

                if step_tag_id is not None:
                    idx_cpu = (token_id_cpu == step_tag_id).nonzero(as_tuple=True)[0]
                    idx = async_tensor_h2d(idx_cpu, data.device)
                    data = data[idx]

                pooled_data.append(data)

        return pooled_data


def get_tok_pooling_method(pooling_type: TokenPoolingType | str):
    if pooling_type == "ALL":
        return AllPool()
    if pooling_type == "STEP":
        return StepPool()

    raise NotImplementedError(f"Unknown tokenwise pooling type: {pooling_type!r}")
