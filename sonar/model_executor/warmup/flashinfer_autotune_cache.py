# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer autotune cache helpers."""

import hashlib
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

import sonar.envs as envs
from sonar.config import CacheConfig, SonarConfig, replace
from sonar.logger import init_logger

if TYPE_CHECKING:
    from sonar.distributed.parallel_state import GroupCoordinator
    from sonar.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


def _normalize_cache_config_for_hash(cache_config: CacheConfig) -> CacheConfig:
    """Normalize cache config for hashing"""
    if cache_config.mamba_block_size is None and cache_config.kv_cache_layout is None:
        return cache_config
    normalized = replace(cache_config, mamba_block_size=None)
    assert normalized.kv_cache_layout is None
    return normalized


def flashinfer_autotune_cache_hash(runner: "GPUModelRunner") -> str:
    sonar_config: SonarConfig = runner.sonar_config
    cache_config = _normalize_cache_config_for_hash(sonar_config.cache_config)
    if cache_config is not sonar_config.cache_config:
        sonar_config = replace(sonar_config, cache_config=cache_config)
    config_hash = sonar_config.compute_hash(include_version=False)
    return hashlib.sha256(config_hash.encode()).hexdigest()


def flashinfer_autotune_root() -> Path:
    """Root directory of the FlashInfer autotune cache (no runner needed)."""
    override_dir = envs.SONAR_FLASHINFER_AUTOTUNE_CACHE_DIR
    if override_dir:
        return Path(override_dir).expanduser()
    from flashinfer.jit import env as flashinfer_jit_env

    flashinfer_workspace = flashinfer_jit_env.FLASHINFER_WORKSPACE_DIR
    return (
        Path(envs.SONAR_CACHE_ROOT)
        / "flashinfer_autotune_cache"
        / flashinfer_workspace.parent.name
        / flashinfer_workspace.name
    )


def flashinfer_autotune_filename(dp_rank: int, rank: int) -> str:
    """Per-rank cache filename, shared by the engine reader and D2D exchange."""
    return f"autotune_configs_dp{dp_rank}_rank{rank}.json"


def resolve_flashinfer_autotune_file(runner: "GPUModelRunner") -> Path:
    output_dir = flashinfer_autotune_root() / flashinfer_autotune_cache_hash(runner)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "autotune_configs.json"


def resolve_rank_flashinfer_autotune_file(runner: "GPUModelRunner") -> Path:
    """The per-rank variant of :func:`resolve_flashinfer_autotune_file`."""
    from sonar.distributed.parallel_state import get_world_group

    path = resolve_flashinfer_autotune_file(runner)
    dp_rank = runner.sonar_config.parallel_config.data_parallel_rank
    return path.with_name(flashinfer_autotune_filename(dp_rank, get_world_group().rank_in_group))


def use_flashinfer_autotune_v2(runner: "GPUModelRunner") -> bool:
    from sonar.distributed.parallel_state import get_node_count, get_world_group
    from sonar.utils.flashinfer import has_flashinfer_autotune_v2

    # Elastic EP transfers tuning state to joining workers with v1
    # save_configs/load_configs, which do not export managed-store winners.
    if runner.sonar_config.parallel_config.enable_elastic_ep:
        return False
    # PP stages profile independently. Keep their group-scoped v1 files until
    # managed-cache reads and publishes can be isolated across tuning groups.
    if runner.sonar_config.parallel_config.pipeline_parallel_size > 1:
        return False
    if not has_flashinfer_autotune_v2():
        return False
    # Use the initialized topology: Ray may span nodes with config.nnodes=1.
    # A cache hit skips profiling collectives, so node-local stores with
    # different contents cannot safely participate in synchronized tuning.
    if get_node_count() != 1:
        logger.info_once(
            "Using legacy FlashInfer autotune cache synchronization for "
            "multi-node deployments; managed-cache synchronization is "
            "currently supported only within one node."
        )
        return False
    # Lift this gate after pinning a release with the fix below and validating
    # distributed managed-cache tuning: https://github.com/flashinfer-ai/flashinfer/pull/6185
    if get_world_group().world_size > 1:
        logger.info_once(
            "Using legacy FlashInfer autotune cache synchronization for "
            "multi-rank deployments until FlashInfer coordinates cache hits."
        )
        return False
    return True


def resolve_flashinfer_autotune_v2_root() -> Path | None:
    """Placement root for FlashInfer's managed autotune store.

    Unlike :func:`resolve_flashinfer_autotune_file`, no Sonar-side identity
    (config hash, flashinfer version, arch) is folded into the path: the
    managed store keys entries per op and hashes the software/hardware
    environment itself. ``None`` selects FlashInfer's default root.
    """
    override_dir = envs.SONAR_FLASHINFER_AUTOTUNE_CACHE_DIR
    return Path(override_dir).expanduser() if override_dir else None


def sync_flashinfer_autotune_cache(
    runner: "GPUModelRunner",
    group: "GroupCoordinator",
) -> None:
    cache: bytes | str | None = None
    if group.rank_in_group == 0 and runner.sonar_config.kernel_config.enable_flashinfer_autotune:
        try:
            from sonar.platforms import current_platform
            from sonar.utils.flashinfer import has_flashinfer

            if has_flashinfer() and current_platform.has_device_capability(90):
                from flashinfer.autotuner import AutoTuner

                with tempfile.TemporaryDirectory() as temp_dir:
                    path = Path(temp_dir) / "autotune_configs.json"
                    AutoTuner.get().save_configs(str(path))
                    cache = path.read_bytes()
        except Exception as exc:
            cache = f"{type(exc).__name__}: {exc}"

    cache = group.broadcast_object(cache)
    if isinstance(cache, str):
        raise RuntimeError(f"Failed to serialize FlashInfer autotune state: {cache}")
    if cache is None or group.rank_in_group == 0:
        return

    from flashinfer.autotuner import AutoTuner

    with tempfile.NamedTemporaryFile() as f:
        f.write(cache)
        f.flush()
        if not AutoTuner.get().load_configs(f.name):
            raise RuntimeError("FlashInfer autotune cache is incompatible")


def write_flashinfer_autotune_cache(cache_path: Path, contents: bytes) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=cache_path.parent, suffix=".tmp", prefix=f".{cache_path.name}.")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(contents)
        os.replace(tmp_path, cache_path)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp_path)
        raise
