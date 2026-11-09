# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import sys
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file

from sonar.config.load import LoadConfig
from sonar.model_executor.model_loader import nf12_checkpoint as checkpoint
from sonar.model_executor.model_loader.default_loader import DefaultModelLoader


def _marker(path, marker=None):
    (path / "config.json").write_text(json.dumps({"nf12_config": {"version": 1} if marker is None else marker}))


def _components(name="weight"):
    return {
        f"{name}.nf12_packed": torch.ones((16, 96), dtype=torch.uint8),
        f"{name}.nf12_escapes": torch.empty((0, 8), dtype=torch.int16),
        f"{name}.nf12_exp_base": torch.tensor([0], dtype=torch.uint8),
    }


def _iterator(path, **kwargs):
    return checkpoint.nf12_weights_iterator([str(path / "model.safetensors")], **kwargs)


@pytest.mark.parametrize("marker", [{"version": 2}, {"version": True}, {"version": 1.0}, {}, [], "1"])
def test_invalid_marker(tmp_path, marker):
    _marker(tmp_path, marker)
    with pytest.raises(ValueError, match="nf12_config"):
        checkpoint.is_nf12_checkpoint(str(tmp_path), str(tmp_path), None, None, None)


def test_root_marker_in_cached_snapshot(tmp_path, monkeypatch):
    # HF snapshots contain symlinks to blobs; config lookup must not follow a
    # weight symlink into the blob directory or use a nested source's config.
    root = tmp_path / "snapshots" / "commit"
    nested = root / "nested" / "weights"
    nested.mkdir(parents=True)
    _marker(root)
    (nested / "config.json").write_text("{}")
    monkeypatch.setattr(checkpoint, "cached_file", Mock(side_effect=AssertionError("unexpected Hub lookup")))
    assert checkpoint.is_nf12_checkpoint("org/model", str(nested), "nested/weights", "commit", None)


@pytest.mark.parametrize("marked", [False, True])
def test_unencoded_tensors_do_not_import_nf12(tmp_path, monkeypatch, marked):
    if marked:
        _marker(tmp_path)
    tensors = {"packed_experts": torch.arange(16, dtype=torch.uint8), "scale": torch.ones(3, dtype=torch.float32)}
    save_file(tensors, tmp_path / "model.safetensors")
    monkeypatch.setitem(sys.modules, "nf12", None)
    loader = DefaultModelLoader(LoadConfig(load_format="safetensors"))
    actual = dict(loader._get_weights_iterator(DefaultModelLoader.Source(str(tmp_path), None)))
    assert actual.keys() == tensors.keys()
    for name in tensors:
        assert actual[name].dtype == tensors[name].dtype
        torch.testing.assert_close(actual[name], tensors[name], rtol=0, atol=0)


@pytest.mark.parametrize(
    "options,match",
    [
        ({"load_format": "fastsafetensors"}, "load_format"),
        ({"load_format": "instanttensor"}, "load_format"),
        ({"model_loader_extra_config": {"enable_multithread_load": True}}, "multithread"),
        ({"safetensors_load_strategy": "eager"}, "safetensors_load_strategy"),
        ({"safetensors_load_strategy": "prefetch"}, "safetensors_load_strategy"),
        ({"safetensors_load_strategy": "torchao"}, "safetensors_load_strategy"),
    ],
)
def test_incompatible_load_options(tmp_path, options, match):
    _marker(tmp_path)
    save_file(_components(), tmp_path / "model.safetensors")
    loader = DefaultModelLoader(LoadConfig(**options))
    with pytest.raises(ValueError, match=match):
        loader._get_weights_iterator(DefaultModelLoader.Source(str(tmp_path), None))


def test_non_safetensors_marked_checkpoint_rejected(tmp_path):
    _marker(tmp_path)
    torch.save({"weight": torch.ones(1)}, tmp_path / "model.pt")
    loader = DefaultModelLoader(LoadConfig(load_format="pt"))
    with pytest.raises(ValueError, match="load_format"):
        loader._get_weights_iterator(DefaultModelLoader.Source(str(tmp_path), None))


@pytest.mark.parametrize("missing", ["nf12_packed", "nf12_escapes", "nf12_exp_base"])
def test_missing_components(tmp_path, missing):
    tensors = _components()
    del tensors[f"weight.{missing}"]
    save_file(tensors, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="missing components"):
        list(_iterator(tmp_path))


@pytest.mark.parametrize("extra", ["weight", "weight.nf12_other"])
def test_extra_components_and_name_collision(tmp_path, extra):
    tensors = _components()
    tensors[extra] = torch.ones(1)
    save_file(tensors, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="conflicting|Invalid NF12"):
        list(_iterator(tmp_path))


def test_components_must_share_shard(tmp_path):
    tensors = _components()
    save_file({"weight.nf12_packed": tensors.pop("weight.nf12_packed")}, tmp_path / "a.safetensors")
    save_file(tensors, tmp_path / "b.safetensors")
    with pytest.raises(ValueError, match="missing components"):
        list(checkpoint.nf12_weights_iterator([str(tmp_path / "a.safetensors"), str(tmp_path / "b.safetensors")]))


@pytest.mark.parametrize(
    "component,value,match",
    [
        ("nf12_packed", torch.ones((16, 96), dtype=torch.int8), "packed"),
        ("nf12_packed", torch.ones((15, 96), dtype=torch.uint8), "packed"),
        ("nf12_packed", torch.ones((16, 48), dtype=torch.uint8), "packed"),
        ("nf12_packed", torch.empty((0, 96), dtype=torch.uint8), "packed"),
        ("nf12_escapes", torch.empty((0, 8), dtype=torch.bfloat16), "escapes"),
        ("nf12_escapes", torch.empty((0, 7), dtype=torch.int16), "escapes"),
        ("nf12_escapes", torch.empty((129, 8), dtype=torch.int16), "escapes"),
        ("nf12_exp_base", torch.tensor(0, dtype=torch.uint8), "exp_base"),
        ("nf12_exp_base", torch.tensor([0], dtype=torch.int16), "exp_base"),
        ("nf12_exp_base", torch.tensor([121], dtype=torch.uint8), "exp_base"),
    ],
)
def test_invalid_component_metadata(tmp_path, component, value, match):
    tensors = _components()
    tensors[f"weight.{component}"] = value
    save_file(tensors, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match=match):
        list(_iterator(tmp_path))


@pytest.mark.parametrize("group", range(4))
def test_global_escape_validation(group):
    packed = torch.ones((32, 96), dtype=torch.uint8)
    planes = packed.view(-1, 3, 512)
    # Escapes in distinct super-atoms still reference one global table.
    for atom in range(2):
        planes[atom, 0, 4 * group : 4 * group + 4] = 0
        plane, offset = 1 + group // 2, 8 * (group % 2)
        planes[atom, plane, offset : offset + 8] = 0
        planes[atom, plane, offset] = atom
    checkpoint._validate_escape_indices(packed, 2, "weight")
    with pytest.raises(ValueError, match="out of range"):
        checkpoint._validate_escape_indices(packed, 1, "weight")
    with pytest.raises(ValueError, match="escape count"):
        checkpoint._validate_escape_indices(packed, 3, "weight")


def test_expert_filter_skips_before_decode(tmp_path, monkeypatch):
    tensors = _components("model.layers.0.mlp.experts.1.gate_proj.weight")
    tensors["scale"] = torch.ones(1)
    save_file(tensors, tmp_path / "model.safetensors")
    monkeypatch.setattr(checkpoint, "_decode_weight", Mock(side_effect=AssertionError("unexpected decode")))
    assert list(dict(_iterator(tmp_path, local_expert_ids={0}))) == ["scale"]


def test_unused_weight_filter_receives_logical_names(tmp_path, monkeypatch):
    _marker(tmp_path)
    save_file(_components("unused.weight"), tmp_path / "model.safetensors")
    seen = []

    def is_unused(name):
        seen.append(name)
        return name == "unused.weight"

    monkeypatch.setattr(checkpoint, "_decode_weight", Mock(side_effect=AssertionError("unexpected decode")))
    loader = DefaultModelLoader(LoadConfig(load_format="safetensors"))
    source = DefaultModelLoader.Source(str(tmp_path), None, is_unused_weight=is_unused)
    assert list(loader._get_weights_iterator(source)) == []
    assert seen and set(seen) == {"unused.weight"}


@pytest.mark.parametrize("load_format", ["auto", "hf", "safetensors"])
def test_cuda_roundtrip_returns_full_bf16_before_tp(tmp_path, load_format):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("NF12 requires SM80+ CUDA")
    nf12 = pytest.importorskip("nf12")
    generator = torch.Generator(device="cuda").manual_seed(42)
    weight = torch.randn((32, 128), device="cuda", dtype=torch.bfloat16, generator=generator)
    # Escapes on both sides of a prospective TP split, including raw NaN bits.
    special = torch.tensor([0, -32768, 1, -32767, 32640, -128, 32705, -63], dtype=torch.int16, device="cuda")
    weight.view(torch.int16)[0, :8] = special
    weight.view(torch.int16)[-1, -8:] = special
    encoded = nf12.encode(weight)
    assert encoded.num_escapes > 0
    tensors = {name: value.cpu() for name, value in encoded.state_dict("weight.").items()}
    tensors["expert_payload"] = torch.arange(16, dtype=torch.uint8)
    tensors["scale"] = torch.ones(2, dtype=torch.float32)
    save_file(tensors, tmp_path / "model.safetensors")
    _marker(tmp_path)
    loader = DefaultModelLoader(LoadConfig(load_format=load_format))
    source = DefaultModelLoader.Source(str(tmp_path), None, prefix="model.")
    actual = dict(loader._get_weights_iterator(source))
    decoded = actual["model.weight"]
    assert decoded.device.type == "cpu" and decoded.dtype == torch.bfloat16
    assert decoded.shape == weight.shape
    expected = weight.cpu().view(torch.int16)
    torch.testing.assert_close(decoded.view(torch.int16), expected, rtol=0, atol=0)
    torch.testing.assert_close(actual["model.expert_payload"], tensors["expert_payload"], rtol=0, atol=0)
    assert actual["model.scale"].dtype == torch.float32
