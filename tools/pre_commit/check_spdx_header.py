# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


@dataclass(frozen=True)
class HeaderStyle:
    """Comment syntax and preamble handling for an SPDX header."""

    comment_prefix: str
    preserve_shebang: bool = False


class SPDXStatus(Enum):
    EMPTY = "empty"
    COMPLETE = "complete"
    MISSING_LICENSE = "missing_license"
    MISSING_COPYRIGHT = "missing_copyright"
    MISSING_BOTH = "missing_both"


LICENSE_TEXT = "SPDX-License-Identifier: Apache-2.0"
COPYRIGHT_TEXT = "SPDX-FileCopyrightText: Copyright contributors to the vLLM project"
FILE_STYLES = {
    ".py": HeaderStyle("#", preserve_shebang=True),
    ".rs": HeaderStyle("//"),
    ".proto": HeaderStyle("//"),
}


def file_style(file_path: str) -> HeaderStyle:
    """Return the declared header style for a file."""
    suffix = Path(file_path).suffix
    try:
        return FILE_STYLES[suffix]
    except KeyError:
        raise ValueError(f"Unsupported file type: {file_path}") from None


def spdx_header(style: HeaderStyle) -> tuple[str, str]:
    """Return the SPDX header for a file style."""
    license_line = f"{style.comment_prefix} {LICENSE_TEXT}"
    copyright_line = f"{style.comment_prefix} {COPYRIGHT_TEXT}"
    return license_line, copyright_line


def header_insertion_index(style: HeaderStyle, lines: list[str]) -> int:
    """Return the line index where a missing header should be inserted."""
    if style.preserve_shebang and lines and lines[0].startswith("#!"):
        return 1
    return 0


def check_spdx_header_status(file_path: str) -> SPDXStatus:
    license_line, copyright_line = spdx_header(file_style(file_path))
    with open(file_path, encoding="utf-8") as file:
        lines = file.readlines()
        if not lines:
            return SPDXStatus.EMPTY
        has_license = False
        has_copyright = False

        for line in lines:
            stripped = line.strip()
            if stripped == license_line:
                has_license = True
            elif stripped == copyright_line:
                has_copyright = True

        if has_license and has_copyright:
            return SPDXStatus.COMPLETE
        if has_license:
            return SPDXStatus.MISSING_COPYRIGHT
        if has_copyright:
            return SPDXStatus.MISSING_LICENSE
        return SPDXStatus.MISSING_BOTH


def add_header(file_path: str, status: SPDXStatus) -> None:
    style = file_style(file_path)
    license_line, copyright_line = spdx_header(style)
    full_spdx_header = f"{license_line}\n{copyright_line}\n"
    with open(file_path, "r+", encoding="utf-8") as file:
        lines = file.readlines()
        file.seek(0, 0)
        file.truncate()

        if status == SPDXStatus.MISSING_BOTH:
            insertion_index = header_insertion_index(style, lines)
            file.writelines(lines[:insertion_index])
            file.write(full_spdx_header)
            remaining_lines = lines[insertion_index:]
            if remaining_lines and remaining_lines[0].strip():
                file.write("\n")
            file.writelines(remaining_lines)
            return

        if status == SPDXStatus.MISSING_COPYRIGHT:
            for i, line in enumerate(lines):
                if line.strip() == license_line:
                    lines.insert(i + 1, f"{copyright_line}\n")
                    break
            file.writelines(lines)
            return

        if status == SPDXStatus.MISSING_LICENSE:
            for i, line in enumerate(lines):
                if line.strip() == copyright_line:
                    lines.insert(i, f"{license_line}\n")
                    break
            file.writelines(lines)


def main() -> int:
    files_to_fix: list[tuple[str, SPDXStatus]] = []
    for file_path in sys.argv[1:]:
        status = check_spdx_header_status(file_path)
        if status in {
            SPDXStatus.MISSING_BOTH,
            SPDXStatus.MISSING_COPYRIGHT,
            SPDXStatus.MISSING_LICENSE,
        }:
            files_to_fix.append((file_path, status))

    if files_to_fix:
        print("The following files are missing the SPDX header:")
        for file_path, status in files_to_fix:
            print(f"  {file_path}")
            add_header(file_path, status)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
