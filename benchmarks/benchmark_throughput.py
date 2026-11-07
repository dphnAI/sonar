# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys

if __name__ == "__main__":
    print("""DEPRECATED: This script has been moved to the Sonar CLI.

Please use the following command instead:
    sonar bench throughput

For help with the new command, run:
    sonar bench throughput --help

Alternatively, you can run the new command directly with:
    python -m sonar.entrypoints.cli.main bench throughput --help
""")
    sys.exit(1)
