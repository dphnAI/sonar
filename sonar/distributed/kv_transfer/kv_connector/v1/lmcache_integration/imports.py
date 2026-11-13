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


class _SchedulerContract(ast.NodeTransformer):
    """Migrate maintained LMCache adapters to the V2 scheduler's data model."""

    def __init__(self):
        self.needs_token_tensor = False

    @staticmethod
    def _resumed_test(node: ast.AST, operator: type[ast.cmpop]) -> bool:
        return (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.ops[0], operator)
            and isinstance(node.left, ast.Name)
            and node.left.id in ("req_id", "request_id")
            and len(node.comparators) == 1
            and ast.dump(node.comparators[0]) == ast.dump(ast.parse("cached_reqs.resumed_req_ids", mode="eval").body)
        )

    def visit_If(self, node: ast.If):
        if self._resumed_test(node.test, ast.NotIn):
            # Resumed requests now arrive in scheduled_new_reqs; every cached
            # request appends its blocks. Only unwrap the known MP append guard.
            if (
                not node.orelse
                and len(node.body) == 1
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Call)
                and ast.dump(node.body[0].value.func)
                == ast.dump(ast.parse("request_tracker.append_block_ids", mode="eval").body)
            ):
                return [self.visit(statement) for statement in node.body]
            raise RuntimeError("Unsupported LMCache cached-request append guard")
        if (
            isinstance(node.test, ast.Call)
            and isinstance(node.test.func, ast.Name)
            and node.test.func.id == "hasattr"
            and len(node.test.args) == 2
            and isinstance(node.test.args[0], ast.Name)
            and node.test.args[0].id == "cached_reqs"
            and isinstance(node.test.args[1], ast.Constant)
            and node.test.args[1].value == "resumed_req_ids"
        ):
            if (
                len(node.body) == 1
                and isinstance(node.body[0], ast.Assign)
                and len(node.body[0].targets) == 1
                and isinstance(node.body[0].targets[0], ast.Name)
                and node.body[0].targets[0].id == "preempted"
                and self._resumed_test(node.body[0].value, ast.In)
            ):
                node.body[0].value = ast.Constant(value=False)
                return node.body[0]
            raise RuntimeError("Unsupported LMCache cached-request preemption branch")
        return self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript):
        self.generic_visit(node)
        if (
            isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "new_request"
            and node.value.attr == "prompt_token_ids"
        ):
            # On resume the prefill includes generated tokens, not just prompt.
            node.value.attr = "prefill_token_ids"
        return node

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch"
            and node.func.attr == "tensor"
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], ast.Attribute)
            and isinstance(node.args[0].value, ast.Name)
            and node.args[0].value.id == "request"
            and node.args[0].attr == "prompt_token_ids"
        ):
            self.needs_token_tensor = True
            node.func = ast.Name(id="_sonar_token_ids_to_tensor", ctx=ast.Load())
            node.args[0] = ast.BoolOp(op=ast.Or(), values=[node.args[0], ast.List(elts=[], ctx=ast.Load())])
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "copy"
            and not node.args
            and not node.keywords
            and isinstance(node.func.value, ast.Subscript)
            and isinstance(node.func.value.value, ast.Attribute)
            and node.func.value.value.attr == "prefill_token_ids"
        ):
            return node.func.value
        return node


def _migrate_scheduler_contract(tree: ast.Module, fullname: str) -> ast.Module:
    if fullname not in (
        "lmcache.integration.vllm.lmcache_mp_connector",
        "lmcache.integration.vllm.lmcache_mp_metadata",
        "lmcache.integration.vllm.vllm_v1_adapter",
        "lmcache.inprocess.integration.vllm.vllm_v1_adapter",
    ):
        return tree
    migration = _SchedulerContract()
    tree = migration.visit(tree)
    if migration.needs_token_tensor:
        # Keep the module docstring and any future imports first.
        index = 0
        for statement in tree.body:
            if (
                isinstance(statement, ast.ImportFrom)
                and statement.module == "__future__"
                or isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            ):
                index += 1
            else:
                break
        tree.body.insert(
            index,
            ast.ImportFrom(
                module="sonar.utils.torch_utils",
                names=[ast.alias(name="token_ids_to_tensor", asname="_sonar_token_ids_to_tensor")],
                level=0,
            ),
        )
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "resumed_req_ids":
            raise RuntimeError(f"Unsupported removed scheduler field in installed {fullname}: resumed_req_ids")
    return tree


class _LMCacheLoader(SourceFileLoader):
    def get_code(self, fullname):
        # Do not load or overwrite the installed .pyc files.
        source = self.get_data(self.path)
        tree = _SonarImports().visit(ast.parse(source, filename=self.path))
        tree = _migrate_scheduler_contract(tree, fullname)
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
