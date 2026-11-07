# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import os
import sys
from pathlib import Path

import aphrodite.envs as envs

JSON_FORMAT = "%(asctime)s %(levelname)s %(name)s %(processName)s %(process)d %(message)s"


class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    DEBUG = "\033[36m"
    INFO = "\033[1m\033[36m"
    WARNING = "\033[1m\033[33m"
    ERROR = "\033[1m\033[31m"
    CRITICAL = "\033[1m\033[41m\033[37m"

    TIME = "\033[2m"
    PATH = "\033[2m\033[34m"


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False
    return os.environ.get("TERM", "") != "dumb"


class NewLineFormatter(logging.Formatter):
    """Adds logging prefix to newlines to align multiline log messages."""

    def __init__(self, fmt, datefmt=None, style="%", *, log_level: str | None = None):
        super().__init__(fmt, datefmt, style)

        self.use_relpath = log_level == "DEBUG"
        if self.use_relpath:
            self.root_dir = Path(__file__).resolve().parent.parent.parent

        self.use_color = _supports_color() and os.environ.get("APHRODITE_LOGGING_COLOR", "1") in ("1", "true", "True")
        self.verbose_logging = envs.APHRODITE_LOGGING_VERBOSE
        self.level_colors = {
            "DEBUG": Colors.DEBUG,
            "INFO": Colors.INFO,
            "WARNING": Colors.WARNING,
            "ERROR": Colors.ERROR,
            "CRITICAL": Colors.CRITICAL,
        }

    def format(self, record):
        if self.verbose_logging:
            original_datefmt = self.datefmt
            self.datefmt = "%m-%d %H:%M:%S"
        else:
            original_datefmt = self.datefmt
            self.datefmt = "%H:%M:%S"

        def shrink_path(relpath: Path) -> str:
            parts = list(relpath.parts)
            new_parts = []
            if parts and parts[0] == "aphrodite":
                parts = parts[1:]
            if parts and parts[0] == "v1":
                new_parts += parts[:2]
                parts = parts[2:]
            elif parts:
                new_parts += parts[:1]
                parts = parts[1:]
            if len(parts) > 2:
                new_parts += ["..."] + parts[-2:]
            else:
                new_parts += parts
            return "/".join(new_parts)

        if self.use_relpath:
            abs_path = getattr(record, "pathname", None)
            if abs_path:
                try:
                    relpath = Path(abs_path).resolve().relative_to(self.root_dir)
                except Exception:
                    relpath = Path(record.filename)
            else:
                relpath = Path(record.filename)
            record.fileinfo = shrink_path(relpath)
        else:
            record.fileinfo = record.filename

        if record.fileinfo.endswith(".py"):
            record.fileinfo = record.fileinfo[:-3]

        max_fileinfo_width = 15
        if len(record.fileinfo) > max_fileinfo_width:
            record.fileinfo = "..." + record.fileinfo[-(max_fileinfo_width - 3) :]

        if self.verbose_logging:
            msg = super().format(record)
        else:
            original_fmt = self._style._fmt
            from aphrodite.logger import _FORMAT_INFO

            self._style._fmt = _FORMAT_INFO
            msg = super().format(record)
            self._style._fmt = original_fmt

        if "WARNING" in msg:
            msg = msg.replace("WARNING", "WARN", 1)

        if self.use_color:
            level_color = self.level_colors.get(record.levelname, "")
            level_str = "WARN" if record.levelname == "WARNING" else record.levelname

            if level_str in msg:
                msg = msg.replace(level_str, f"{level_color}{level_str}{Colors.RESET}", 1)

            asctime = self.formatTime(record, self.datefmt)
            if asctime in msg:
                msg = msg.replace(asctime, f"{Colors.TIME}{asctime}{Colors.RESET}", 1)

            if self.verbose_logging:
                fileinfo_str = f"[{record.fileinfo:<15}:{record.lineno:>4}]"
                if fileinfo_str in msg:
                    msg = msg.replace(
                        fileinfo_str,
                        f"{Colors.PATH}{fileinfo_str}{Colors.RESET}",
                        1,
                    )

        self.datefmt = original_datefmt

        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\r\n" + parts[0])
        return msg


class ColoredFormatter(NewLineFormatter):
    """Compatibility alias for older logger configuration."""


class UvicornFormatter(logging.Formatter):
    """Uvicorn formatter that matches Aphrodite's console logging style."""

    def __init__(self, fmt, datefmt=None, style="%"):
        super().__init__(fmt, datefmt, style)

        self.verbose_logging = envs.APHRODITE_LOGGING_VERBOSE
        self.use_color = _supports_color() and os.environ.get("APHRODITE_LOGGING_COLOR", "1") in ("1", "true", "True")
        self.level_colors = {
            "DEBUG": Colors.DEBUG,
            "INFO": Colors.INFO,
            "WARNING": Colors.WARNING,
            "ERROR": Colors.ERROR,
            "CRITICAL": Colors.CRITICAL,
        }
        self.path_color = Colors.PATH
        self.time_color = Colors.TIME
        self.reset_color = Colors.RESET

    def format(self, record):
        if not self.verbose_logging:
            original_datefmt = self.datefmt
            self.datefmt = "%H:%M:%S"
        else:
            original_datefmt = self.datefmt

        msg = super().format(record)

        if not self.verbose_logging:
            self.datefmt = original_datefmt

        if "WARNING" in msg:
            msg = msg.replace("WARNING", "WARN", 1)

        if self.use_color:
            level_color = self.level_colors.get(record.levelname, "")
            level_str = "WARN" if record.levelname == "WARNING" else record.levelname

            if level_str in msg:
                msg = msg.replace(level_str, f"{level_color}{level_str}{self.reset_color}", 1)

            asctime = self.formatTime(
                record,
                self.datefmt if not self.verbose_logging else "%m-%d %H:%M:%S",
            )
            if asctime in msg:
                msg = msg.replace(asctime, f"{self.time_color}{asctime}{self.reset_color}", 1)

            if self.verbose_logging:
                name_with_lineno = f"[{record.name:<15}:{record.lineno:>4}]"
                if name_with_lineno in msg:
                    msg = msg.replace(
                        name_with_lineno,
                        f"{self.path_color}{name_with_lineno}{self.reset_color}",
                        1,
                    )

        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\r\n" + parts[0])

        return msg
