# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()


class EngineCapacity(BaseModel):
    dp_rank: int
    max_model_len: int
    max_num_seqs: int
    kv_cache_capacity_tokens: int | None
    estimated_concurrency_at_max_model_len: float | None


class CapacityResponse(BaseModel):
    schema_version: int = 1
    scope: str = "connected_engines"
    engines: list[EngineCapacity]


@router.get("/v1/capacity", response_model=CapacityResponse)
async def capacity(request: Request) -> CapacityResponse:
    """Return startup cache capacity for connected DP engines, not live availability.

    Token capacity is a group-aware equivalent at max_model_len. Concurrency
    estimates exclude scheduler limits and are not admission guarantees.
    Null cache values indicate that cache capacity is unavailable or inapplicable.
    """
    client = getattr(request.app.state, "engine_client", None)
    core = getattr(client, "engine_core", None)
    reports = getattr(core, "capacity_reports", {})
    ranks = [int.from_bytes(identity, "little") for identity in getattr(core, "core_engines", [])]
    if not ranks or any(rank not in reports for rank in ranks):
        raise HTTPException(status_code=503, detail="Engine capacity is not available")
    return CapacityResponse(
        engines=[
            EngineCapacity(
                dp_rank=rank,
                max_model_len=reports[rank].max_model_len,
                max_num_seqs=reports[rank].max_num_seqs,
                kv_cache_capacity_tokens=reports[rank].kv_cache_size_tokens,
                estimated_concurrency_at_max_model_len=reports[rank].kv_cache_max_concurrency,
            )
            for rank in sorted(ranks)
        ]
    )
