# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Losslessly encode a local safetensors checkpoint, one shard per GPU worker.

python tools/convert_nf12_checkpoint.py --model /models/Kimi-K3.1-Pro \
    --output /models/Kimi-K3.1-Pro-NF12 --devices 0,1,2,3,4,5,6,7

The output must not exist. Failed conversions leave a named staging directory
for inspection, never a checkpoint at the requested output path.
"""

import argparse
import json
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, fields
from pathlib import Path
from threading import Event

import torch
from safetensors import safe_open
from safetensors.torch import save_file

INDEX_NAME = "model.safetensors.index.json"
COMPONENT_SUFFIXES = (".nf12_packed", ".nf12_escapes", ".nf12_exp_base")
WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf"}


@dataclass
class Counts:
    tensors: int = 0
    eligible: int = 0
    encoded: int = 0
    original_bytes: int = 0
    stored_bytes: int = 0
    encoded_original_bytes: int = 0
    encoded_bytes: int = 0
    escapes: int = 0

    def add(self, other):
        for field in fields(self):
            setattr(self, field.name, getattr(self, field.name) + getattr(other, field.name))

    def summary(self):
        return (
            f"tensors={self.tensors} eligible={self.eligible} encoded={self.encoded} "
            f"unchanged={self.tensors - self.encoded} "
            f"original_bytes={self.original_bytes} stored_bytes={self.stored_bytes} "
            f"saved_bytes={self.original_bytes - self.stored_bytes} "
            f"encoded_original_bytes={self.encoded_original_bytes} "
            f"encoded_bytes={self.encoded_bytes} escapes={self.escapes}"
        )


def _eligible(tensor):
    return (
        tensor.dtype == torch.bfloat16
        and tensor.ndim == 2
        and tensor.shape[0] > 0
        and tensor.shape[1] > 0
        and tensor.shape[0] % 16 == 0
        and tensor.shape[1] % 64 == 0
    )


def _encode_tensor(name, tensor, device):
    import nf12

    weight = tensor.to(device)
    encoded = nf12.encode(weight)
    decoded = encoded.decode()
    # Compare representations, not floats: NaNs, signed zero and subnormals count.
    if (
        decoded.dtype != torch.bfloat16
        or decoded.shape != weight.shape
        or not torch.equal(decoded.view(torch.int16), weight.view(torch.int16))
    ):
        raise ValueError(f"NF12 decoding did not recover the original BF16 values for {name}")
    state = encoded.state_dict(name + ".")
    size = sum(value.nbytes for value in state.values())
    if size >= tensor.nbytes:
        return None, 0
    return {key: value.cpu() for key, value in state.items()}, encoded.num_escapes


def _atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def _convert_shard(source, staging, shard, expected_keys, device, stop):
    counts = Counts()
    tensors = {}
    with safe_open(source / shard, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        if expected_keys is not None and keys != expected_keys:
            raise ValueError(
                f"Index does not match {shard}: missing={expected_keys - keys}, extra={keys - expected_keys}"
            )
        if any(key.endswith(COMPONENT_SUFFIXES) for key in keys):
            raise ValueError(f"{shard} already contains reserved NF12 component names")
        metadata = handle.metadata()
        for name in sorted(keys):
            if stop.is_set():
                raise RuntimeError("Conversion cancelled after another worker failed")
            tensor = handle.get_tensor(name)
            counts.tensors += 1
            counts.original_bytes += tensor.nbytes
            state = None
            if _eligible(tensor):
                counts.eligible += 1
                try:
                    state, escapes = _encode_tensor(name, tensor, device)
                except Exception as error:
                    raise RuntimeError(f"Encoding {name} in {shard} on {device} failed: {error}") from error
            if state is None:
                tensors[name] = tensor
                counts.stored_bytes += tensor.nbytes
            else:
                size = sum(value.nbytes for value in state.values())
                tensors.update(state)
                counts.encoded += 1
                counts.encoded_original_bytes += tensor.nbytes
                counts.encoded_bytes += size
                counts.stored_bytes += size
                counts.escapes += escapes
        if stop.is_set():
            raise RuntimeError("Conversion cancelled after another worker failed")
        destination = staging / shard
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        save_file(tensors, temporary, metadata=metadata)
        os.replace(temporary, destination)
    print(f"[{device}] {shard}: {counts.summary()}", flush=True)
    return dict.fromkeys(tensors, shard), counts


def _read_layout(source):
    index_path = source / INDEX_NAME
    if index_path.is_file():
        with index_path.open() as stream:
            index = json.load(stream)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"{index_path} must contain a nonempty weight_map")
        shards = {}
        for name, filename in weight_map.items():
            if not isinstance(filename, str):
                raise ValueError(f"Invalid shard name for {name}: {filename!r}")
            relative = Path(filename)
            if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".safetensors":
                raise ValueError(f"Unsafe or unsupported shard path: {filename!r}")
            if relative.as_posix() != filename:
                raise ValueError(f"Noncanonical shard path: {filename!r}")
            if not (source / relative).is_file():
                raise ValueError(f"Missing source shard: {filename}")
            shards.setdefault(filename, set()).add(name)
        return index, shards
    shards = {path.name: None for path in sorted(source.glob("*.safetensors"))}
    if not shards:
        raise ValueError(f"No safetensors checkpoint found in {source}")
    return {"metadata": {}, "weight_map": {}}, shards


def _copy_assets(source, staging):
    def ignore(directory, names):
        return [
            name
            for name in names
            if name in {".git", ".cache"}
            or (Path(directory) == source and name == "config.json")
            or Path(name).suffix in WEIGHT_SUFFIXES
            or name.endswith((".safetensors.index.json", ".bin.index.json"))
        ]

    shutil.copytree(source, staging, dirs_exist_ok=True, ignore=ignore)


def convert_checkpoint(model, output, devices=(0,)):
    source, destination = Path(model).resolve(), Path(output).resolve()
    if not source.is_dir():
        raise ValueError(f"Source is not a directory: {source}")
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Source and output must be distinct, non-overlapping directories")
    if destination.exists():
        raise ValueError(f"Output already exists; refusing to overwrite: {destination}")
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("Provide one or more distinct CUDA device indices")
    for device in devices:
        if not isinstance(device, int) or device < 0 or device >= torch.accelerator.device_count():
            raise ValueError(f"Unavailable CUDA device: {device}")
        if torch.cuda.get_device_capability(device)[0] < 8:
            raise ValueError(f"NF12 requires SM80 or newer: cuda:{device}")
    with (source / "config.json").open() as stream:
        config = json.load(stream)
    if "nf12_config" in config:
        raise ValueError("Source already has nf12_config; supply the original checkpoint")
    index, shards = _read_layout(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.partial-", dir=destination.parent))
    stop = Event()
    totals = Counts()
    weight_map = {}
    shard_names = sorted(shards)
    workers = min(len(devices), len(shard_names))

    def run_worker(device_index, assigned):
        device = torch.device("cuda", device_index)
        torch.accelerator.set_device_index(device_index)
        results = []
        with torch.inference_mode():
            for shard in assigned:
                results.append(_convert_shard(source, staging, shard, shards[shard], device, stop))
        return results

    try:
        _copy_assets(source, staging)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_worker, devices[i], shard_names[i::workers]) for i in range(workers)]
            try:
                for future in as_completed(futures):
                    for mapping, counts in future.result():
                        duplicate = weight_map.keys() & mapping.keys()
                        if duplicate:
                            raise ValueError(f"Duplicate checkpoint tensor keys: {sorted(duplicate)}")
                        weight_map.update(mapping)
                        totals.add(counts)
            except BaseException:
                stop.set()
                raise
        index["weight_map"] = dict(sorted(weight_map.items()))
        index.setdefault("metadata", {})["total_size"] = totals.stored_bytes
        config["nf12_config"] = {"version": 1}
        _atomic_json(staging / INDEX_NAME, index)
        _atomic_json(staging / "config.json", config)
        if destination.exists():
            raise ValueError(f"Output appeared during conversion; refusing to overwrite: {destination}")
        os.rename(staging, destination)
    except BaseException as error:
        raise RuntimeError(
            f"NF12 conversion failed: {error}. Incomplete files are in {staging}; "
            "do not load that directory. Remove it before retrying. Source files were not modified."
        ) from error
    print(f"Completed {len(shards)} shards -> {destination}: {totals.summary()}", flush=True)
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, type=Path, help="Local source checkpoint directory (read-only)")
    parser.add_argument("--output", "-o", required=True, type=Path, help="New output directory; must not exist")
    parser.add_argument(
        "--devices", default="0", help="Comma-separated visible CUDA indices; one bounded shard worker per GPU"
    )
    args = parser.parse_args()
    try:
        devices = tuple(int(device.strip()) for device in args.devices.split(","))
        convert_checkpoint(args.model, args.output, devices)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
