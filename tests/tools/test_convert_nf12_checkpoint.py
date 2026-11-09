# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from tools.convert_nf12_checkpoint import INDEX_NAME, _eligible, _read_layout, convert_checkpoint


@pytest.mark.parametrize(
    "shape,dtype,expected",
    [
        ((16, 64), torch.bfloat16, True),
        ((0, 64), torch.bfloat16, False),
        ((16, 0), torch.bfloat16, False),
        ((15, 64), torch.bfloat16, False),
        ((16, 63), torch.bfloat16, False),
        ((1, 16, 64), torch.bfloat16, False),
        ((64,), torch.bfloat16, False),
        ((16, 64), torch.float32, False),
        ((16, 64), torch.uint8, False),
    ],
)
def test_nf12_checkpoint_eligibility(shape, dtype, expected):
    assert _eligible(torch.empty(shape, dtype=dtype)) is expected


@pytest.mark.parametrize("filename", ["../weights.safetensors", "/weights.safetensors", "weights.bin"])
def test_nf12_checkpoint_rejects_unsafe_shard_paths(tmp_path, filename):
    (tmp_path / INDEX_NAME).write_text(json.dumps({"weight_map": {"weight": filename}}))
    with pytest.raises(ValueError, match="Unsafe or unsupported"):
        _read_layout(tmp_path)


def test_nf12_checkpoint_rejects_overlapping_paths(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for output in (source, source / "nested", tmp_path):
        with pytest.raises(ValueError, match="non-overlapping"):
            convert_checkpoint(source, output)
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        convert_checkpoint(source, output)


@pytest.fixture
def nf12_codec():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("NF12 requires CUDA SM80 or newer")
    return pytest.importorskip("nf12")


def _digest_files(directory):
    return {
        path.relative_to(directory): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*")
        if path.is_file()
    }


def test_nf12_checkpoint_mixed_round_trip(tmp_path, nf12_codec):
    source, output = tmp_path / "source", tmp_path / "encoded"
    source.mkdir()
    config = {"model_type": "test", "quantization_config": {"quant_method": "compressed-tensors", "config_groups": {}}}
    (source / "config.json").write_text(json.dumps(config))
    (source / "tokenizer.json").write_text('{"version": "1.0"}')
    (source / "modeling_custom.py").write_text("MODEL_VERSION = 1\n")
    (source / "assets").mkdir()
    (source / "assets" / "vocab.txt").write_text("hello\n")
    weight = torch.ones((32, 128), dtype=torch.bfloat16)
    weight.view(torch.int16)[0, :8] = torch.tensor([0, -32768, 1, -32767, 32640, -128, 32705, -63], dtype=torch.int16)
    # All BF16 bit patterns: too many escapes for a storage benefit.
    wide = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).reshape(256, 256)
    originals = {
        "matrix.weight": weight,
        "wide.weight": wide,
        "quantized.weight": torch.arange(64, dtype=torch.uint8).reshape(8, 8),
        "scale": torch.tensor([0, -2147483648, 2143289345], dtype=torch.int32).view(torch.float32),
        "unsupported.weight": torch.ones((3, 7), dtype=torch.bfloat16),
    }
    shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    save_file(
        {name: value for name, value in originals.items() if name != "wide.weight"},
        source / shards[0],
        metadata={"format": "pt", "custom": "preserved"},
    )
    save_file({"wide.weight": wide}, source / shards[1])
    index = {
        "metadata": {"total_size": sum(value.nbytes for value in originals.values()), "custom": "preserved"},
        "weight_map": {name: shards[1] if name == "wide.weight" else shards[0] for name in originals},
    }
    (source / INDEX_NAME).write_text(json.dumps(index))
    before = _digest_files(source)
    totals = convert_checkpoint(source, output, (torch.cuda.current_device(),))
    assert _digest_files(source) == before
    assert json.loads((output / "config.json").read_text()) == {**config, "nf12_config": {"version": 1}}
    for name in ("tokenizer.json", "modeling_custom.py", "assets/vocab.txt"):
        assert (output / name).read_bytes() == (source / name).read_bytes()
    result_index = json.loads((output / INDEX_NAME).read_text())
    physical = {}
    for shard in shards:
        state = load_file(output / shard)
        physical.update(state)
        assert all(result_index["weight_map"][name] == shard for name in state)
    with safe_open(output / shards[0], framework="pt") as handle:
        assert handle.metadata() == {"format": "pt", "custom": "preserved"}
    assert set(result_index["weight_map"]) == set(physical)
    assert result_index["metadata"] == {
        "total_size": sum(value.nbytes for value in physical.values()),
        "custom": "preserved",
    }
    encoded = nf12_codec.NF12Weight.from_state_dict(physical, "matrix.weight.").to("cuda")
    assert torch.equal(encoded.decode().cpu().view(torch.int16), weight.view(torch.int16))
    for name, value in originals.items():
        if name != "matrix.weight":
            assert physical[name].dtype == value.dtype
            assert physical[name].shape == value.shape
            assert torch.equal(physical[name].view(torch.uint8), value.view(torch.uint8))
    assert totals.encoded == 1
    assert totals.eligible == 2
    assert totals.escapes > 0
    assert totals.original_bytes == index["metadata"]["total_size"]
    assert totals.stored_bytes == result_index["metadata"]["total_size"]
    assert totals.stored_bytes < totals.original_bytes
    assert not list(tmp_path.glob(".encoded.partial-*"))


def test_nf12_checkpoint_failure_stays_unpublished(tmp_path, nf12_codec):
    source, output = tmp_path / "source", tmp_path / "encoded"
    source.mkdir()
    (source / "config.json").write_text("{}")
    save_file({"actual": torch.ones(1)}, source / "model.safetensors")
    (source / INDEX_NAME).write_text(json.dumps({"weight_map": {"missing": "model.safetensors"}}))
    before = _digest_files(source)
    with pytest.raises(RuntimeError, match="Index does not match.*Incomplete files are in"):
        convert_checkpoint(source, output, (torch.cuda.current_device(),))
    assert not output.exists()
    assert _digest_files(source) == before
    (partial,) = tmp_path.glob(".encoded.partial-*")
    assert not (partial / "config.json").exists()
    assert not (partial / INDEX_NAME).exists()
