# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


def test_qwen3_5_models_use_platform_specific_implementation():
    from sonar.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5MTP
    from sonar.platforms import current_platform

    # Only the NVIDIA variant drops torch.compile; ROCm and other platforms
    # keep the previous torch.compile implementation in
    # sonar/model_executor/models/qwen3_5{,_mtp}.py.
    if current_platform.is_cuda():
        model_module = "sonar.models.qwen3_5.nvidia.model"
        mtp_module = "sonar.models.qwen3_5.nvidia.mtp"
    else:
        model_module = "sonar.model_executor.models.qwen3_5"
        mtp_module = "sonar.model_executor.models.qwen3_5_mtp"
    assert Qwen3_5ForCausalLM.__module__ == model_module
    assert Qwen3_5MTP.__module__ == mtp_module


def test_qwen3_5_nvidia_models_do_not_use_torch_compile_wrapper():
    # Import the NVIDIA modules directly so the assertion is independent of
    # the platform this test runs on.
    from sonar.compilation.wrapper import TorchCompileWithNoGuardsWrapper
    from sonar.models.qwen3_5.nvidia import model as model_module
    from sonar.models.qwen3_5.nvidia import mtp as mtp_module

    assert not issubclass(model_module.Qwen3_5Model, TorchCompileWithNoGuardsWrapper)
    assert not issubclass(mtp_module.Qwen3_5MultiTokenPredictor, TorchCompileWithNoGuardsWrapper)
    assert not issubclass(mtp_module.Qwen3_5MTP, TorchCompileWithNoGuardsWrapper)


def test_qwen3_5_legacy_models_keep_torch_compile_wrapper():
    # The legacy implementation in sonar/model_executor/models/ serves ROCm and
    # other non-NVIDIA platforms; it must keep the torch.compile wrapper.
    from sonar.compilation.wrapper import TorchCompileWithNoGuardsWrapper
    from sonar.model_executor.models import qwen3_5 as model_module
    from sonar.model_executor.models import qwen3_5_mtp as mtp_module

    assert issubclass(model_module.Qwen3_5Model, TorchCompileWithNoGuardsWrapper)
    assert issubclass(mtp_module.Qwen3_5MultiTokenPredictor, TorchCompileWithNoGuardsWrapper)
    assert issubclass(mtp_module.Qwen3_5MTP, TorchCompileWithNoGuardsWrapper)
