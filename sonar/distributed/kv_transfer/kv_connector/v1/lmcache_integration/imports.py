# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the Sonar project
"""Adapt LMCache's engine imports without modifying its installed sources."""

import ast
import sys
from importlib.abc import MetaPathFinder
from importlib.machinery import PathFinder, SourceFileLoader

_MODULES = {
    "vllm.attention": "sonar.model_executor.layers.attention.attention",
    "vllm.attention.backends.abstract": "sonar.v1.attention.backend",
}
_SYMBOLS = {
    "VllmConfig": "SonarConfig",
    "get_current_vllm_config": "get_current_sonar_config",
}
_ATTRIBUTES = {
    "VLLM_RPC_BASE_PATH": "SONAR_RPC_BASE_PATH",
    "_vllm_config": "_sonar_config",
}


def _module_name(name: str) -> str:
    if name in _MODULES:
        return _MODULES[name]
    if name == "vllm" or name.startswith("vllm."):
        return "sonar" + name[4:]
    return name


class _SonarImports(ast.NodeTransformer):
    def visit_ImportFrom(self, node: ast.ImportFrom):
        if not node.level and node.module and _module_name(node.module) != node.module:
            node.module = _module_name(node.module)
            for alias in node.names:
                if alias.name in _SYMBOLS:
                    alias.asname = alias.asname or alias.name
                    alias.name = _SYMBOLS[alias.name]
        return node

    def visit_Import(self, node: ast.Import):
        roots = []
        for alias in node.names:
            name = _module_name(alias.name)
            if name != alias.name:
                if not alias.asname:
                    if "." not in alias.name:
                        alias.asname = alias.name
                    else:
                        # Dotted imports bind the root package, not the leaf module.
                        roots.append(ast.Import(names=[ast.alias(name="sonar", asname="vllm")]))
                alias.name = name
        return [node, *roots]

    def visit_Constant(self, node: ast.Constant):
        # LMCache's hashers import module names stored in lists at runtime.
        if isinstance(node.value, str) and node.value.startswith("vllm."):
            node.value = _module_name(node.value)
        return node

    def visit_Attribute(self, node: ast.Attribute):
        self.generic_visit(node)
        node.attr = _ATTRIBUTES.get(node.attr, node.attr)
        return node


class _LMCacheLoader(SourceFileLoader):
    def get_code(self, fullname):
        # Do not load or overwrite the installed .pyc files.
        source = self.get_data(self.path)
        tree = _SonarImports().visit(ast.parse(source, filename=self.path))
        return compile(ast.fix_missing_locations(tree), self.path, "exec", dont_inherit=True)


class _LMCacheFinder(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "lmcache" and not fullname.startswith("lmcache."):
            return None
        spec = PathFinder.find_spec(fullname, path)
        if spec is not None and isinstance(spec.loader, SourceFileLoader):
            spec.loader = _LMCacheLoader(fullname, spec.loader.path)
        return spec


def enable_sonar_imports() -> None:
    if not any(isinstance(finder, _LMCacheFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _LMCacheFinder())
