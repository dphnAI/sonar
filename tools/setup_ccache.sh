#!/bin/bash

# Sonar Engine ccache setup script
# This script configures ccache for optimal performance when building Sonar Engine

# Change to project root directory
cd "$(dirname "$0")/.."

echo "Setting up ccache for Sonar Engine development..."

# Configure ccache with optimal settings
bin/micromamba run -r conda -n sonar-runtime ccache --set-config=max_size=10G
bin/micromamba run -r conda -n sonar-runtime ccache --set-config=compression=true
bin/micromamba run -r conda -n sonar-runtime ccache --set-config=compression_level=6
bin/micromamba run -r conda -n sonar-runtime ccache --set-config=sloppiness=file_macro,locale,time_macros

# Show current configuration
echo "Current ccache configuration:"
bin/micromamba run -r conda -n sonar-runtime ccache --show-config | grep -E "(cache_dir|max_size|compression)"

# Show current stats
echo ""
echo "Current ccache statistics:"
bin/micromamba run -r conda -n sonar-runtime ccache --show-stats

echo ""
echo "ccache setup complete!"
echo ""
echo "To build with ccache enabled, simply run:"
echo "  ./runtime.sh pip install -ve . --force-reinstall"
echo ""
echo "To check ccache performance:"
echo "  ./runtime.sh ccache --show-stats"
echo ""
echo "To clear ccache:"
echo "  ./runtime.sh ccache --clear" 