#!/bin/bash

# Sonar Engine ccache status checker

# Change to project root directory
cd "$(dirname "$0")/.."

echo "=== ccache Status ==="
echo "ccache location: $(./runtime.sh which ccache)"
echo "ccache version: $(./runtime.sh ccache --version | head -1)"
echo ""

echo "=== Configuration ==="
./runtime.sh ccache --show-config | grep -E "(cache_dir|max_size|compression)"
echo ""

echo "=== Statistics ==="
./runtime.sh ccache --show-stats
echo ""

echo "=== Build System Detection ==="
echo "The build system will automatically detect and use ccache when available."
echo "You can verify this by watching for 'CMAKE_*_COMPILER_LAUNCHER=ccache' in build logs." 