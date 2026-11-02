#!/bin/bash

wget -qO- https://github.com/mamba-org/micromamba-releases/releases/download/1.5.8-0/micromamba-linux-64.tar.bz2 | tar -xvj bin/micromamba
if [ ! -f "conda/envs/aphrodite-runtime/bin/python" ]; then
 bin/micromamba create --no-shortcuts -r conda -n aphrodite-runtime -f environment.yaml -y
fi
bin/micromamba create --no-shortcuts -r conda -n aphrodite-runtime -f environment.yaml -y
#bin/micromamba install -r conda -n aphrodite-runtime gxx=10 -c conda-forge -y
bin/micromamba run -r conda -n aphrodite-runtime pip install -r requirements/cuda.txt
# Make it so the correct NVCC is found. Looks only within the current working
# directory, since find will return the *first* result, leading to conflicts
# if you have multiple environments and one of them does not contain CUDA runtime.
CUDA_HOME=$(realpath "$(find ./ -type d -path "*/conda/envs/aphrodite-runtime" 2>/dev/null | head -n 1)")
export CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
bin/micromamba run -r conda -n aphrodite-runtime pip install -ve .