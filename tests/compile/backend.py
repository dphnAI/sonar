# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from copy import deepcopy

from torch import fx
from torch._ops import OpOverload, OpOverloadPacket
from torch.fx._utils import lazy_format_graph_code

from sonar.compilation import depyf_compat
from sonar.compilation.passes.fx_utils import find_op_nodes
from sonar.compilation.passes.inductor_pass import (
    InductorPass,
    pass_context,
)
from sonar.compilation.passes.ir.inplace_functionalization import (
    SonarIRInplaceFunctionalizationPass,
)
from sonar.compilation.passes.pass_manager import with_pattern_match_debug
from sonar.compilation.passes.sonar_inductor_pass import SonarInductorPass
from sonar.config import SonarConfig, get_current_sonar_config
from sonar.config.utils import Range
from sonar.logger import init_logger

logger = init_logger("sonar.tests.compile.backend")


class LazyInitPass(InductorPass):
    """If there's a pass that we want to initialize lazily in a test,
    we can wrap it in LazyInitPass, which will initialize the pass when invoked
    and then immediately invoke it.
    """

    def __init__(self, pass_cls: type[SonarInductorPass], sonar_config: SonarConfig):
        self.pass_cls = pass_cls
        self.sonar_config = weakref.proxy(sonar_config)  # avoid cycle

    def __call__(self, graph: fx.Graph) -> None:
        self.pass_ = self.pass_cls(self.sonar_config)
        self.pass_(graph)


class TestBackend:
    """This class provides a simple Inductor backend that can be used for testing.
    It takes a list of custom passes and runs them after Inductor's passes.
    It also saves the graph before and after the custom passes for inspection.

    Inductor config can be modified directly by editing the inductor_config
    property. This can be helpful for adding passes like the
    'pre_grad_custom_pass' and the 'post_grad_custom_pre_pass'.
    Inductor config is default-initialized from SonarConfig.CompilationConfig.
    """

    def __init__(self, *passes: InductorPass | Callable[[fx.Graph], None]):
        self.custom_passes = list(passes)
        sonar_config = get_current_sonar_config()
        compile_config = sonar_config.compilation_config
        self.range = Range(1, sonar_config.scheduler_config.max_num_batched_tokens)
        # Deepcopy to allow multiple TestBackend instances to use the same SonarConfig
        self.inductor_config = deepcopy(compile_config.inductor_compile_config)
        self.inductor_config["force_disable_caches"] = True
        self.inductor_config["post_grad_custom_post_pass"] = self.post_pass

        # Add SonarIRInplaceFunctionalizationPass as pre-grad pass by default
        self.inductor_config["pre_grad_custom_pass"] = SonarIRInplaceFunctionalizationPass(sonar_config)

        if debug_dump_path := sonar_config.compile_debug_dump_path():
            logger.debug("Dumping depyf output to %s", debug_dump_path)
            self.debug_ctx = depyf_compat.prepare_debug(debug_dump_path.as_posix())
        else:
            self.debug_ctx = nullcontext()

    def __call__(self, graph: fx.GraphModule, example_inputs):
        self.graph_pre_compile = deepcopy(graph)
        from torch._inductor.compile_fx import compile_fx

        with self.debug_ctx, pass_context(self.range):
            return compile_fx(graph, example_inputs, config_patches=self.inductor_config)

    @with_pattern_match_debug
    def post_pass(self, graph: fx.Graph):
        self.graph_pre_pass = deepcopy(graph)
        lazy_format_graph_code("graph_pre_pass", graph.owning_module)

        SonarInductorPass.dump_prefix = 0
        for pass_ in self.custom_passes:
            pass_(graph)
            SonarInductorPass.dump_prefix += 1

        SonarInductorPass.dump_prefix = None

        self.graph_post_pass = deepcopy(graph)
        lazy_format_graph_code("graph_post_pass", graph.owning_module)
        # assign by reference, will reflect the final state of the graph
        self.final_graph = graph

    def check_before_ops(self, ops: Sequence[OpOverload | OpOverloadPacket], fully_replaced=True):
        for op in ops:
            num_pre = len(list(find_op_nodes(op, self.graph_pre_pass)))
            num_post = len(list(find_op_nodes(op, self.graph_post_pass)))
            assert num_pre > 0, f"Op {op.name()} not found in pre-pass graph"
            assert num_pre > num_post, f"All nodes remain for op {op.name()}"
            if fully_replaced:
                assert num_post == 0, f"Unexpected op {op.name()} in post-pass graph"

    def check_after_ops(self, ops: Sequence[OpOverload | OpOverloadPacket]):
        for op in ops:
            num_pre = len(list(find_op_nodes(op, self.graph_pre_pass)))
            num_post = len(list(find_op_nodes(op, self.graph_post_pass)))
            assert num_pre == 0, f"Unexpected op {op.name()} in pre-pass graph"
            assert num_post > 0, f"Op {op.name()} not found in post-pass graph"

    def op_count(self, op: OpOverload | OpOverloadPacket, before=False) -> int:
        graph = self.graph_pre_pass if before else self.graph_post_pass
        return len(list(find_op_nodes(op, graph)))

    def print_graphs(self):
        print("=== Graph before custom passes ===")
        print(self.graph_pre_pass.python_code(root_module="self", verbose=True).src)
        print("=== Graph after custom passes ===")
        print(self.graph_post_pass.python_code(root_module="self", verbose=True).src)
