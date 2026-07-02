# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib

from torch import nn

from aphrodite.config import ModelConfig, AphroditeConfig
from aphrodite.config.load import LoadConfig
from aphrodite.model_executor.model_loader.base_loader import BaseModelLoader
from aphrodite.tracing import instrument

_MODELEXPRESS_LOADER_MODULE = "modelexpress.engines.aphrodite.loader"
_MISSING_MODELEXPRESS_MODULES = frozenset(
    {
        "modelexpress",
        "modelexpress.engines",
        "modelexpress.engines.aphrodite",
        _MODELEXPRESS_LOADER_MODULE,
    }
)


def _missing_modelexpress_error() -> ImportError:
    return ImportError(
        "The 'modelexpress' load format requires the ModelExpress Python package. "
        "Install it with `pip install modelexpress`."
    )


class ModelExpressModelLoader(BaseModelLoader):
    """Thin Aphrodite loader wrapper for ModelExpress."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        self._loader = self._load_modelexpress_loader(load_config)

    @staticmethod
    def _load_modelexpress_loader(load_config: LoadConfig) -> BaseModelLoader:
        try:
            module = importlib.import_module(_MODELEXPRESS_LOADER_MODULE)
        except ModuleNotFoundError as exc:
            if exc.name not in _MISSING_MODELEXPRESS_MODULES:
                raise
            raise _missing_modelexpress_error() from exc

        ModelExpressAphroditeLoader = module.MxModelLoader
        return ModelExpressAphroditeLoader(load_config)

    def download_model(self, model_config: ModelConfig) -> None:
        self._loader.download_model(model_config)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        self._loader.load_weights(model, model_config)

    @instrument(span_name="Load model")
    def load_model(
        self,
        aphrodite_config: AphroditeConfig,
        model_config: ModelConfig,
        prefix: str = "",
    ) -> nn.Module:
        model = self._loader.load_model(
            aphrodite_config=aphrodite_config,
            model_config=model_config,
            prefix=prefix,
        )
        return model.eval()
