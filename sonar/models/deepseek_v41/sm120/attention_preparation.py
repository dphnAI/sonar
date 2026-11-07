# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight-stage preparation for V4.1 rotary, scaling and cache writers."""

from dataclasses import replace

import torch
from b12x.attention import compressed_sparse_mla as mla
from b12x.attention.compressed_sparse_mla import rotary, weight_scale

from aphrodite.utils.b12x import B12xPreparationUnit, B12xWorkload


class AttentionHelperPreparation:
    """Loaded-resource owner, independent of unpublished attention caches."""

    def __init__(self, attention):
        self.attention = attention

    def get_b12x_preparation_units(self, layer: object, workload: B12xWorkload) -> tuple[B12xPreparationUnit, ...]:
        from b12x.preparation import FrozenMapping, MemoryRequirements, PreparedCall

        attn = self.attention
        table = attn.rotary_emb.cos_sin_cache
        device = table.device
        attn._declare_attention(device)
        roles = [("kv", 1, 512, 1), ("q", attn.n_local_heads, 512, 1)]
        if attn.indexer is not None:
            roles.append(("index_query", attn.indexer.heads, 128, 1))
        if attn.compressor is not None:
            roles.extend(
                (
                    ("index_key", 1, 128, attn.compress_ratio),
                    ("latent", 1, 512, attn.compress_ratio),
                )
            )
        plans: dict[str, object] = {}
        requests = []
        for role, heads, dim, ratio in roles:
            query = rotary.Query(
                max_rows=attn.capacity,
                heads=heads,
                dim=dim,
                ratio=ratio,
                cos_sin_dtype=str(table.dtype).removeprefix("torch."),
            )
            declaration = rotary.plan(query, device=device)
            if role == "kv":
                native_memory = declaration._memory_requirements

                def memory(config, detected, native_memory=native_memory):
                    return MemoryRequirements.sequential(
                        (
                            native_memory(config, detected),
                            attn._staging_memory(),
                        )
                    )

                declaration = replace(
                    declaration,
                    _memory_requirements=memory,
                    invocation=FrozenMapping(
                        {
                            "aphrodite_staging": tuple(
                                (name, shape, str(dtype)) for name, shape, dtype in attn._staging_specs
                            )
                        }
                    ),
                )
            plans[role] = declaration

            def prepare(state, heads=heads, dim=dim):
                attn._prepare(device)
                source = torch.ones((1, heads, dim), dtype=torch.bfloat16, device=device)
                output = torch.empty_like(source)
                positions = torch.zeros(1, dtype=torch.int64, device=device)
                return PreparedCall(run=lambda: state.run(source, positions, table, out=output))

            requests.append(
                declaration.request(
                    name=f"{attn.prefix}.helper.{role}",
                    prepare_call=prepare,
                )
            )
        if attn.indexer is not None:
            declaration = weight_scale.plan(
                weight_scale.Query(max_elements=attn.capacity * attn.indexer.heads),
                device=device,
            )
            plans["index_weights"] = declaration

            def prepare_weights(state):
                source = torch.ones((1, attn.indexer.heads), dtype=torch.bfloat16, device=device)
                output = torch.empty_like(source)
                return PreparedCall(run=lambda: state.run(source, out=output))

            requests.append(
                declaration.request(
                    name=f"{attn.prefix}.helper.index_weights",
                    prepare_call=prepare_weights,
                )
            )
        cache_roles = [("swa_cache_write", attn.swa_cache_layer.block_size, "swa")]
        if attn.compressor is not None:
            cache_roles.append(("indexed_cache_write", attn._main_page, "indexed"))
        for role, page_size, kind in cache_roles:
            declaration = mla.plan_cache_writer(
                mla.CacheWriterQuery(
                    max_rows=attn.capacity,
                    page_size=page_size,
                    cache_kind=kind,
                    slot_dtype="int64",
                ),
                device=device,
            )
            plans[role] = declaration

            def prepare_writer(state, page_size=page_size, kind=kind):
                source = torch.ones((1, 512), dtype=torch.bfloat16, device=device)
                cache = torch.empty(
                    (
                        1,
                        mla.page_nbytes(page_size, cache_kind=kind, cache_format="deepseek_v41"),
                    ),
                    dtype=torch.uint8,
                    device=device,
                )
                slots = torch.zeros(1, dtype=torch.int64, device=device)
                return PreparedCall(run=lambda: state.run(source, cache, slots))

            requests.append(
                declaration.request(
                    name=f"{attn.prefix}.helper.{role}",
                    prepare_call=prepare_writer,
                )
            )
        attn._helper_plans = plans
        return (
            B12xPreparationUnit(
                name="V41AttentionHelpers",
                key=(attn.prefix, attn.capacity),
                requests=tuple(requests),
                stage="weights",
            ),
            attn._wo_preparation_unit(workload),
        )
