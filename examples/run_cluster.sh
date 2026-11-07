#!/bin/bash
#
# Launch a Ray cluster inside Docker for Sonar inference.
#
# This script can start either a head node or a worker node, depending on the
# --head or --worker flag provided as the third positional argument.
#
# Usage:
# 1. Designate one machine as the head node and execute:
#    bash run_cluster.sh \
#         alpindale/aphrodite-openai \
#         <head_node_ip> \
#         --head \
#         /abs/path/to/huggingface/cache
#
# 2. On every worker machine, execute:
#    bash run_cluster.sh \
#         alpindale/aphrodite-openai \
#         <head_node_ip> \
#         --worker \
#         /abs/path/to/huggingface/cache
# 
# The script automatically detects and sets:
#   - SONAR_HOST_IP (detected from network interface)
#   - GLOO_SOCKET_IFNAME (detected network interface name)
#   - NCCL_SOCKET_IFNAME (detected network interface name)
#
# You can override these by passing -e flags, e.g.:
#   -e SONAR_HOST_IP=<your_node_ip> -e GLOO_SOCKET_IFNAME=eth0
# Keep each terminal session open. Closing a session stops the associated Ray
# node and thereby shuts down the entire cluster.
# Every machine must be reachable at the supplied IP address.
#
# The container is named "node-<random_suffix>". To open a shell inside
# a container after launch, use:
#       docker exec -it node-<random_suffix> /bin/bash
#
# Then, you can execute Sonar commands on the Ray cluster as if it were a
# single machine, e.g. sonar run ...
#
# To stop the container, use:
#       docker stop node-<random_suffix>

# Check for minimum number of required arguments.
if [ $# -lt 4 ]; then
    echo "Usage: $0 docker_image head_node_ip --head|--worker path_to_hf_home [additional_args...]"
    exit 1
fi

# Extract the mandatory positional arguments and remove them from $@.
DOCKER_IMAGE="$1"
HEAD_NODE_ADDRESS="$2"
NODE_TYPE="$3"  # Should be --head or --worker.
PATH_TO_HF_HOME="$4"
shift 4

# Preserve any extra arguments so they can be forwarded to Docker.
ADDITIONAL_ARGS=("$@")

# Validate the NODE_TYPE argument.
if [ "${NODE_TYPE}" != "--head" ] && [ "${NODE_TYPE}" != "--worker" ]; then
    echo "Error: Node type must be --head or --worker"
    exit 1
fi

# Auto-detect network interface and IP for Gloo/NCCL
# This helps avoid connection issues in multi-node setups
detect_network_config() {
    local target_ip="$1"
    
    # Get local IP address
    local local_ip
    if command -v hostname &> /dev/null; then
        local_ip=$(hostname -I | awk '{print $1}')
    else
        # Fallback: get IP from default route interface
        local_ip=$(ip -4 route get 8.8.8.8 2>/dev/null | grep -oP 'src \K[^\s]+' | head -1)
    fi
    
    # Find the network interface for the target IP
    local interface
    if [ "${target_ip}" = "${local_ip}" ] || ip addr show 2>/dev/null | grep -q "${target_ip}"; then
        # Local IP - find interface directly
        interface=$(ip -4 addr show 2>/dev/null | grep -B 2 "${target_ip}" | grep -oP '^\d+:\s+\K[^:]+' | head -1)
    else
        # Remote IP - use routing
        interface=$(ip -4 route get "${target_ip}" 2>/dev/null | grep -oP 'dev \K[^\s]+' | head -1)
    fi
    
    # If interface detection failed, try to get default interface
    if [ -z "${interface}" ] || [ "${interface}" = "lo" ]; then
        # Get default route interface
        interface=$(ip -4 route show default 2>/dev/null | grep -oP 'dev \K[^\s]+' | head -1)
    fi
    
    # If still no interface, try common non-loopback interfaces
    if [ -z "${interface}" ] || [ "${interface}" = "lo" ]; then
        for iface in eth0 ens3 enp0s8 enp6s18; do
            if ip link show "${iface}" &> /dev/null; then
                interface="${iface}"
                break
            fi
        done
    fi
    
    echo "${interface}:${local_ip}"
}

# Detect network configuration
if [ "${NODE_TYPE}" == "--head" ]; then
    # For head node, use the head_node_ip to detect interface
    NETWORK_CONFIG=$(detect_network_config "${HEAD_NODE_ADDRESS}")
    DETECTED_INTERFACE=$(echo "${NETWORK_CONFIG}" | cut -d':' -f1)
    DETECTED_IP="${HEAD_NODE_ADDRESS}"
else
    # For worker node, detect interface for connecting to head node
    # but use worker's own IP (detected from local interface)
    NETWORK_CONFIG=$(detect_network_config "${HEAD_NODE_ADDRESS}")
    DETECTED_INTERFACE=$(echo "${NETWORK_CONFIG}" | cut -d':' -f1)
    DETECTED_IP=$(echo "${NETWORK_CONFIG}" | cut -d':' -f2)
fi

# Use detected values or allow override via environment variables
NODE_IP="${SONAR_HOST_IP:-${DETECTED_IP}}"
SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${DETECTED_INTERFACE}}"

# Warn if we couldn't detect properly
if [ -z "${SOCKET_IFNAME}" ] || [ "${SOCKET_IFNAME}" = "lo" ]; then
    echo "Warning: Could not auto-detect network interface. Using: ${SOCKET_IFNAME:-unknown}"
    echo "  Set GLOO_SOCKET_IFNAME environment variable to override (e.g., eth0, ens3)"
fi

if [ -z "${NODE_IP}" ] || [[ "${NODE_IP}" =~ ^127\. ]]; then
    echo "Warning: Detected IP appears to be loopback: ${NODE_IP}"
    echo "  Set SONAR_HOST_IP environment variable to override"
fi

# Prepare environment variables to pass to Docker
AUTO_ENV_VARS=(
    "-e" "SONAR_HOST_IP=${NODE_IP}"
    "-e" "GLOO_SOCKET_IFNAME=${SOCKET_IFNAME}"
    "-e" "NCCL_SOCKET_IFNAME=${SOCKET_IFNAME}"
)

echo "Auto-detected network configuration:"
echo "  SONAR_HOST_IP=${NODE_IP}"
echo "  GLOO_SOCKET_IFNAME=${SOCKET_IFNAME}"
echo "  NCCL_SOCKET_IFNAME=${SOCKET_IFNAME}"
echo ""

# Generate a unique container name with random suffix.
# Docker container names must be unique on each host.
# The random suffix allows multiple Ray containers to run simultaneously on the same machine,
# for example, on a multi-GPU machine.
CONTAINER_NAME="node-${RANDOM}"

# Define a cleanup routine that removes the container when the script exits.
# This prevents orphaned containers from accumulating if the script is interrupted.
cleanup() {
    docker stop "${CONTAINER_NAME}"
    docker rm "${CONTAINER_NAME}"
}
trap cleanup EXIT

# Build the Ray start command based on the node role.
# The head node manages the cluster and accepts connections on port 6379, 
# while workers connect to the head's address.
RAY_START_CMD="ray start --block"
if [ "${NODE_TYPE}" == "--head" ]; then
    RAY_START_CMD+=" --head --port=6379"
else
    RAY_START_CMD+=" --address=${HEAD_NODE_ADDRESS}:6379"
fi

# Launch the container with the assembled parameters.
# --network host: Allows Ray nodes to communicate directly via host networking
# --shm-size 10.24g: Increases shared memory
# --gpus all: Gives container access to all GPUs on the host
# -v HF_HOME: Mounts HuggingFace cache to avoid re-downloading models
docker run \
    --entrypoint /bin/bash \
    --network host \
    --name "${CONTAINER_NAME}" \
    --shm-size 10.24g \
    --gpus all \
    -v "${PATH_TO_HF_HOME}:/root/.cache/huggingface" \
    "${AUTO_ENV_VARS[@]}" \
    "${ADDITIONAL_ARGS[@]}" \
    "${DOCKER_IMAGE}" -c "${RAY_START_CMD}"
