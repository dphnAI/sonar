# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NF12 storage decoding, before ordinary model weight loading and TP slicing."""

import json
from collections.abc import Callable, Generator
from pathlib import Path

import torch
from safetensors import safe_open
from transformers.utils import cached_file

from sonar.model_executor.model_loader.ep_weight_filter import should_skip_weight

_COMPONENTS = ("nf12_packed", "nf12_escapes", "nf12_exp_base")


def is_nf12_checkpoint(
    model_or_path: str,
    hf_folder: str,
    subfolder: str | None,
    revision: str | None,
    cache_dir: str | None,
) -> bool:
    # hf_folder is the resolved snapshot, possibly with a source subfolder.
    # Do not resolve symlinks: HF snapshot files point into the blob directory.
    root = Path(hf_folder)
    if subfolder:
        for _ in Path(subfolder).parts:
            root = root.parent
    config_path = root / "config.json"
    if not config_path.is_file():
        if Path(model_or_path).is_dir():
            return False
        resolved = cached_file(
            model_or_path,
            "config.json",
            revision=revision,
            cache_dir=cache_dir,
            _raise_exceptions_for_missing_entries=False,
        )
        if resolved is None:
            return False
        config_path = Path(resolved)
    with config_path.open() as file:
        config = json.load(file)
    if "nf12_config" not in config:
        return False
    marker = config["nf12_config"]
    if not isinstance(marker, dict) or type(marker.get("version")) is not int or marker["version"] != 1:
        raise ValueError(f"Unsupported NF12 checkpoint nf12_config: {marker!r}; expected version 1")
    return True


def nf12_logical_name(name: str) -> str:
    """Map physical storage names to the names understood by model filters."""
    base, separator, component = name.rpartition(".")
    if separator and component.startswith("nf12_"):
        if not base or component not in _COMPONENTS:
            raise ValueError(f"Invalid NF12 checkpoint component: {name!r}")
        return base
    return name


def _validate_shapes(file, name: str) -> None:
    packed, escapes, exp_base = (file.get_slice(f"{name}.{component}") for component in _COMPONENTS)
    p, e, b = packed.get_shape(), escapes.get_shape(), exp_base.get_shape()
    if packed.get_dtype() != "U8" or len(p) != 2 or p[0] <= 0 or p[0] % 16 or p[1] <= 0 or p[1] % 96:
        raise ValueError(f"NF12 {name}: packed must be uint8 [N, K*3/2], N % 16 == K % 64 == 0")
    if escapes.get_dtype() != "I16" or len(e) != 2 or e[1] != 8 or e[0] > p[0] * p[1] // 12:
        raise ValueError(f"NF12 {name}: escapes must be int16 [E, 8], E <= N*K/8")
    if exp_base.get_dtype() != "U8" or b != [1]:
        raise ValueError(f"NF12 {name}: exp_base must be uint8 [1]")


def _validate_escape_indices(packed: torch.Tensor, num_escapes: int, name: str) -> None:
    # Tiled v1 has three 512-byte planes per 16x64 super-atom. Plane 0
    # contains four metadata words per lane; planes 1/2 contain their payloads.
    # Inspect bounded CPU views before CUDA decode, whose escape reads are
    # unchecked. Indices are global to the full matrix, never TP-local.
    atoms = packed.numpy().reshape(-1, 3, 512)
    count = 0
    for start in range(0, len(atoms), 4096):
        chunk = atoms[start : start + 4096]
        meta = chunk[:, 0].view("<u4").reshape(-1, 32, 4)
        for group in range(4):
            escaped = meta[:, :, group] == 0
            payload = chunk[:, 1 + group // 2].reshape(-1, 32, 2, 8)
            indices = payload[:, :, group % 2, :4].view("<u4")[..., 0][escaped]
            count += indices.size
            if indices.size and int(indices.max()) >= num_escapes:
                raise ValueError(f"NF12 {name}: escape index out of range for {num_escapes} escapes")
    if count != num_escapes:
        raise ValueError(f"NF12 {name}: escape count {count} does not match escapes tensor ({num_escapes})")


def _decode_weight(file, name: str, device_packed: torch.Tensor | None = None) -> torch.Tensor:
    packed, escapes, exp_base = (file.get_tensor(f"{name}.{component}") for component in _COMPONENTS)
    base = int(exp_base.item())
    if not 0 <= base <= 120:
        raise ValueError(f"NF12 {name}: exp_base must be in [0, 120], got {base}")
    _validate_escape_indices(packed, escapes.shape[0], name)
    if not torch.cuda.is_available():
        raise RuntimeError("Loading an NF12 checkpoint requires a local CUDA device")
    # Ordinary checkpoints never import the optional NF12 CUDA package.
    from nf12 import NF12Weight

    device = torch.device("cuda", torch.accelerator.current_device_index())
    cuda_packed = packed.to(device) if device_packed is None else device_packed
    # Fast shard views can start at arbitrary byte offsets. NF12's vector loads
    # require 16-byte alignment even when the tensor is contiguous.
    if cuda_packed.data_ptr() % 16 or not cuda_packed.is_contiguous():
        cuda_packed = cuda_packed.clone(memory_format=torch.contiguous_format)
    encoded = NF12Weight(cuda_packed, escapes.to(device), base)
    # Decode the entire logical tensor before the existing model loader does TP
    # slicing/fusion/online quantization. Only one decoded tensor is staged.
    # fastsafetensors already owns the device packed tensor; validate against
    # its CPU mmap instead of copying the packed payload back from CUDA.
    decoded = encoded.decode()
    return decoded.cpu() if device_packed is None else decoded


def _validate_shard(file, path: str, seen: set[str]) -> tuple[list[str], set[str]]:
    keys = set(file.keys())
    logical_names = sorted({nf12_logical_name(key) for key in keys})
    encoded_names: set[str] = set()
    # Validate headers before reading any payload, including skipped EP
    # tensors. All three components must live in this same shard.
    for name in logical_names:
        components = {f"{name}.{component}" for component in _COMPONENTS}
        if components & keys:
            if not components <= keys or name in keys:
                raise ValueError(f"NF12 {name}: missing components or conflicting unencoded tensor in {path}")
            _validate_shapes(file, name)
            encoded_names.add(name)
        if name in seen:
            raise ValueError(f"NF12 checkpoint has duplicate logical tensor {name!r}")
        seen.add(name)
    return logical_names, encoded_names


def _fastsafetensors_weights_iterator(
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
    local_expert_ids: set[int] | None,
    is_unused_weight: Callable[[str], bool] | None,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    from sonar.model_executor.model_loader.weight_utils import fastsafetensors_weights_iterator

    if not hf_weights_files:
        return

    seen: set[str] = set()
    encoded_files: dict[str, str] = {}
    for path in hf_weights_files:
        with safe_open(path, framework="pt", device="cpu") as file:
            _, encoded_names = _validate_shard(file, path, seen)
            encoded_files.update((name, path) for name in encoded_names)

    weights = fastsafetensors_weights_iterator(hf_weights_files, use_tqdm_on_load)
    try:
        for physical_name, tensor in weights:
            name = nf12_logical_name(physical_name)
            if should_skip_weight(name, local_expert_ids) or (is_unused_weight is not None and is_unused_weight(name)):
                continue
            if name not in encoded_files:
                yield name, tensor
            elif physical_name == f"{name}.nf12_packed":
                # Physical components need not be adjacent in the stream. Read
                # auxiliary tensors through the mmap, not a pending component
                # dictionary that could retain entire CUDA shards.
                with safe_open(encoded_files[name], framework="pt", device="cpu") as file:
                    yield name, _decode_weight(file, name, device_packed=tensor)
            # Consume every physical tensor on every rank: the underlying
            # iterator broadcasts collectively even when this rank skips it.
    finally:
        weights.close()


def nf12_weights_iterator(
    hf_weights_files: list[str],
    local_expert_ids: set[int] | None = None,
    is_unused_weight: Callable[[str], bool] | None = None,
    *,
    use_fastsafetensors: bool = False,
    use_tqdm_on_load: bool = True,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    if use_fastsafetensors:
        yield from _fastsafetensors_weights_iterator(
            hf_weights_files, use_tqdm_on_load, local_expert_ids, is_unused_weight
        )
        return
    seen: set[str] = set()
    for path in sorted(hf_weights_files):
        with safe_open(path, framework="pt", device="cpu") as file:
            logical_names, encoded_names = _validate_shard(file, path, seen)
            for name in logical_names:
                if should_skip_weight(name, local_expert_ids) or (
                    is_unused_weight is not None and is_unused_weight(name)
                ):
                    continue
                if name in encoded_names:
                    yield name, _decode_weight(file, name)
                else:
                    yield name, file.get_tensor(name)
