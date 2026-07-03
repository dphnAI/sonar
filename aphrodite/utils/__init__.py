# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import uuid
from collections.abc import Iterable

import torch

MASK_64_BITS = (1 << 64) - 1


def random_uuid() -> str:
    return f"{uuid.uuid4().int & MASK_64_BITS:016x}"  # 16 hex chars


def length_from_prompt_token_ids_or_embeds(
    prompt_token_ids: list[int] | torch.Tensor | None,
    prompt_embeds: torch.Tensor | None,
) -> int:
    """Calculate the request length (in number of tokens) give either
    prompt_token_ids or prompt_embeds.
    """
    prompt_token_len = None if prompt_token_ids is None else len(prompt_token_ids)
    prompt_embeds_len = None if prompt_embeds is None else len(prompt_embeds)

    if prompt_token_len is None:
        if prompt_embeds_len is None:
            raise ValueError("Neither prompt_token_ids nor prompt_embeds were defined.")
        return prompt_embeds_len
    else:
        if prompt_embeds_len is not None and prompt_embeds_len != prompt_token_len:
            raise ValueError(
                "Prompt token ids and prompt embeds had different lengths"
                f" prompt_token_ids={prompt_token_len}"
                f" prompt_embeds={prompt_embeds_len}"
            )
        return prompt_token_len


def is_moe_layer(module: torch.nn.Module) -> bool:
    # TODO(bnell): Should use isinstance but can't due to circular dependencies.
    def _check_bases(cls):
        if cls.__name__ == "MoERunnerInterface":
            return True

        for b in cls.__bases__:
            if _check_bases(b):
                return True

    return _check_bases(module.__class__)


def get_progress_log_prefix() -> str:
    """Generate a log-like prefix for progress bars to match Aphrodite logs."""
    import datetime

    from aphrodite import envs
    from aphrodite.logging_utils.formatter import Colors, _supports_color

    verbose_logging = envs.APHRODITE_LOGGING_VERBOSE

    if verbose_logging:
        timestamp = datetime.datetime.now().strftime("%m-%d %H:%M:%S")
        padding = (20 - 3) // 2
        placeholder = " " * padding + "..." + " " * (20 - 3 - padding)

        if _supports_color():
            return (
                f"{Colors.INFO}INFO{Colors.RESET} "
                f"{Colors.TIME}{timestamp}{Colors.RESET} "
                f"{Colors.PATH}[{placeholder}]{Colors.RESET}"
            )
        return f"INFO {timestamp} [{placeholder}]"

    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    if _supports_color():
        return f"{Colors.INFO}INFO{Colors.RESET} {Colors.TIME}{timestamp}{Colors.RESET}"
    return f"INFO {timestamp}"


def tensor_progress_bar(
    iterable: Iterable[tuple[str, torch.Tensor]],
    final_bytes: int | None,
    desc: str = "Processing",
):
    import logging

    from rich.console import Console
    from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

    from aphrodite.distributed.parallel_state import is_global_first_rank

    show_progress = is_global_first_rank()
    if final_bytes is None:
        units = 1024**2
        unit_label = "MB"
        total = None
    elif final_bytes == 0:
        units = 1
        unit_label = "B"
        total = 0
    else:
        unit_power = int(math.log2(final_bytes)) // 10
        units = 1024**unit_power
        unit_labels = {0: "B", 1: "KB", 2: "MB", 3: "GB", 4: "TB", 5: "PB"}
        unit_label = unit_labels.get(unit_power, "B")
        total = final_bytes / units

    if show_progress:
        log_prefix = get_progress_log_prefix()
        console = Console(force_terminal=True)

        root_logger = logging.getLogger()
        original_level = root_logger.level
        root_logger.setLevel(logging.WARNING)

        try:
            with Progress(
                TextColumn(log_prefix + " [progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%") if total is not None else TextColumn(""),
                TextColumn(f"{{task.completed:.2f}}/{{task.total:.2f}} {unit_label}")
                if total is not None
                else TextColumn(f"{{task.completed:.2f}} {unit_label}"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(desc, total=total)
                for item in iterable:
                    if hasattr(item[1], "element_size"):
                        steps = item[1].element_size() * item[1].nelement() / units
                        progress.update(task, advance=steps)
                    yield item
        finally:
            root_logger.setLevel(original_level)
    else:
        yield from iterable
