#!/bin/bash
if [ ! -f "conda/envs/aphrodite-runtime/bin/python" ]; then
./update-runtime.sh
fi
if [ $# -eq 0 ]
  then
    bin/micromamba run -r conda -n aphrodite-runtime bash
    exit
fi
CUDA_HOME=$(realpath "$(find ./ -type d -path "*/conda/envs/aphrodite-runtime" 2>/dev/null | head -n 1)")
export CUDA_HOME
bin/micromamba run -r conda -n aphrodite-runtime "$@"
