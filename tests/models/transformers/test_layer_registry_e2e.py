# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end check of the Transformers backend's hw-agnostic layers.

Serves a tiny Llama through `aphrodite_runner` with and without
`APHRODITE_USE_HW_AGNOSTIC` and checks the layer providers and logprobs. Needs a
real engine, so it runs in the GPU job. The CPU-only unit tests for the
resolution logic (`hw_agnostic.resolve`) are in `test_layer_registry.py`.
"""

import pytest
import torch


@pytest.fixture(scope="module")
def tiny_llama_path(tmp_path_factory):
    """A randomly-initialized microscopic Llama saved to disk (with an ungated
    tokenizer) so Aphrodite can load it like any local checkpoint."""
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        hidden_act="silu",
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config)

    path = tmp_path_factory.mktemp("tiny_llama")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return str(path)


# Registered names of the layers the backend can
# currently route to hw-agnostic implementations.
_COVERED_LAYERS = ("rms_norm", "silu_and_mul")


def _layer_providers(model) -> dict[str, str]:
    """Map each covered layer type present in the model to the provider its
    implementation came from (``hw_agnostic`` or ``aphrodite``).
    """

    def provider_of(module) -> str | None:
        for cls in type(module).__mro__:
            if "hw_agnostic.layers" in cls.__module__:
                return "hw_agnostic"
            if ".model_executor.layers." in cls.__module__:
                return "aphrodite"
        return None

    providers: dict[str, str] = {}
    for module in model.modules():
        name = getattr(module, "name", None)
        if name in _COVERED_LAYERS and name not in providers and (prov := provider_of(module)) is not None:
            providers[name] = prov
    return providers


def _serve(aphrodite_runner, model_path, prompts):
    """Serve the model through the backend; return (layer_providers, logprobs)."""
    with aphrodite_runner(
        model_path,
        model_impl="transformers",
        max_model_len=64,
        enforce_eager=True,
        gpu_memory_utilization=0.3,
    ) as runner:
        assert runner.llm.llm_engine.model_config.using_transformers_backend()
        providers = runner.apply_model(_layer_providers)[0]
        outputs = runner.generate_greedy_logprobs(prompts, max_tokens=32, num_logprobs=5)
        return providers, outputs


def test_hw_agnostic_matches_aphrodite_end_to_end(monkeypatch, aphrodite_runner, tiny_llama_path):
    """Serving the tiny model with hw-agnostic layers matches the Aphrodite baseline."""
    # spawn: `layers` resolves its symbols at import time, so the worker must
    # re-import it with APHRODITE_USE_HW_AGNOSTIC already set (fork would inherit
    # the parent's already-resolved module).
    monkeypatch.setenv("APHRODITE_WORKER_MULTIPROC_METHOD", "spawn")
    # apply_model pickles the introspection function.
    monkeypatch.setenv("APHRODITE_ALLOW_INSECURE_SERIALIZATION", "1")
    from ..utils import check_logprobs_close

    prompts = ["The capital of France is", "Aphrodite is"]

    monkeypatch.setenv("APHRODITE_USE_HW_AGNOSTIC", "0")
    aphrodite_providers, aphrodite_outputs = _serve(aphrodite_runner, tiny_llama_path, prompts)
    # Both replaceable layers present in a Llama block must be Aphrodite's here.
    assert aphrodite_providers == {"rms_norm": "aphrodite", "silu_and_mul": "aphrodite"}

    monkeypatch.setenv("APHRODITE_USE_HW_AGNOSTIC", "1")
    hw_providers, hw_outputs = _serve(aphrodite_runner, tiny_llama_path, prompts)
    assert hw_providers == {"rms_norm": "hw_agnostic", "silu_and_mul": "hw_agnostic"}

    check_logprobs_close(
        outputs_0_lst=aphrodite_outputs,
        outputs_1_lst=hw_outputs,
        name_0="aphrodite",
        name_1="hw_agnostic",
    )
