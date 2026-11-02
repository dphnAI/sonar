#!/bin/bash

# =============================================================================
# Aphrodite Disaggregated Serving Script - P2P NCCL XpYd Architecture
# =============================================================================
# This script demonstrates disaggregated prefill and decode serving using
# P2P NCCL communication. The architecture supports various XpYd configurations:
#
# - 1P3D: 1 Prefill server + 3 Decode servers (current default)
# - 3P1D: 3 Prefill servers + 1 Decode server
# - etc.
#
# Configuration can be customized via environment variables:
#   MODEL: Model to run
#   PREFILL_GPUS: Comma-separated GPU IDs for prefill servers
#   DECODE_GPUS: Comma-separated GPU IDs for decode servers
#   PREFILL_PORTS: Comma-separated ports for prefill servers
#   DECODE_PORTS: Comma-separated ports for decode servers
#   PROXY_PORT: Proxy server port used to setup XpYd connection.
#   TIMEOUT_SECONDS: Server startup timeout
#
# Arguments:
#   --benchmark: Run benchmark after servers start
# =============================================================================

# Parse command line arguments
RUN_BENCHMARK=false
for arg in "$@"; do
    case $arg in
        --benchmark)
            RUN_BENCHMARK=true
            shift
            ;;
        *)
            ;;
    esac
done

# Configuration - can be overridden via environment variables
MODEL=${MODEL:-Qwen/Qwen3-0.6B}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1200}
PROXY_PORT=${PROXY_PORT:-30001}

# Default 1P3D configuration (1 Prefill + 3 Decode)
PREFILL_GPUS=${PREFILL_GPUS:-0}
DECODE_GPUS=${DECODE_GPUS:-1,2,3}
PREFILL_PORTS=${PREFILL_PORTS:-20003}
DECODE_PORTS=${DECODE_PORTS:-20005,20007,20009}

echo "Warning: P2P NCCL disaggregated prefill XpYd support for Aphrodite v1 is experimental and subject to change."
echo ""
echo "Architecture Configuration:"
echo "  Model: $MODEL"
echo "  Prefill GPUs: $PREFILL_GPUS, Ports: $PREFILL_PORTS"
echo "  Decode GPUs: $DECODE_GPUS, Ports: $DECODE_PORTS"
echo "  Proxy Port: $PROXY_PORT"
echo "  Timeout: ${TIMEOUT_SECONDS}s"
echo ""

PIDS=()

# Switch to the directory of the current script
cd "$(dirname "${BASH_SOURCE[0]}")"

check_required_files() {
    local files=("proxy_server.py")
    for file in "${files[@]}"; do
        if [[ ! -f "$file" ]]; then
            echo "Required file $file not found in $(pwd)"
            exit 1
        fi
    done
}

check_num_gpus() {
    # Check if the number of GPUs are >=2 via nvidia-smi
    num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [ "$num_gpus" -lt 2 ]; then
        echo "You need at least 2 GPUs to run disaggregated prefill."
        exit 1
    else
        echo "Found $num_gpus GPUs."
    fi
}

ensure_python_library_installed() {
    echo "Checking if $1 is installed..."
    if ! python3 -c "import $1" > /dev/null 2>&1; then
        echo "$1 is not installed. Please install it via pip install $1."
        exit 1
    else
        echo "$1 is installed."
    fi
}

cleanup() {
    echo "Stopping everything…"
    trap - INT TERM        # prevent re-entrancy
    pkill -9 -f "proxy_server.py"
    kill -- -$$            # negative PID  ==  "this whole process-group"
    wait                   # reap children so we don't leave zombies
    exit 0
}

wait_for_server() {
  local port=$1
  local timeout_seconds=$TIMEOUT_SECONDS
  local start_time=$(date +%s)

  echo "Waiting for server on port $port..."

  while true; do
    if curl -s "localhost:${port}/v1/completions" > /dev/null; then
      echo "Server on port $port is ready."
      return 0
    fi

    local now=$(date +%s)
    if (( now - start_time >= timeout_seconds )); then
      echo "Timeout waiting for server on port $port"
      return 1
    fi

    sleep 1
  done
}

main() {
    check_required_files
    check_num_gpus
    ensure_python_library_installed pandas
    ensure_python_library_installed datasets
    ensure_python_library_installed aphrodite
    ensure_python_library_installed quart

    trap cleanup INT
    trap cleanup USR1
    trap cleanup TERM

    echo "Launching disaggregated serving components..."
    echo "Please check the log files for detailed output:"
    echo "  - prefill*.log: Prefill server logs"
    echo "  - decode*.log: Decode server logs"
    echo "  - proxy.log: Proxy server log"

    # =============================================================================
    # Launch Proxy Server
    # =============================================================================
    echo ""
    echo "Starting proxy server..."
    echo "  - Service Discovery (ZMQ): $PROXY_PORT"
    echo "  - HTTP API: 10001"
    python3 proxy_server.py "$PROXY_PORT" 10001 > proxy.log 2>&1 &
    PIDS+=($!)
    
    # Give proxy a moment to start
    sleep 2

    # Parse GPU and port arrays
    IFS=',' read -ra PREFILL_GPU_ARRAY <<< "$PREFILL_GPUS"
    IFS=',' read -ra DECODE_GPU_ARRAY <<< "$DECODE_GPUS"
    IFS=',' read -ra PREFILL_PORT_ARRAY <<< "$PREFILL_PORTS"
    IFS=',' read -ra DECODE_PORT_ARRAY <<< "$DECODE_PORTS"

    # =============================================================================
    # Launch Prefill Servers (X Producers)
    # =============================================================================
    echo ""
    echo "Starting ${#PREFILL_GPU_ARRAY[@]} prefill server(s)..."
    for i in "${!PREFILL_GPU_ARRAY[@]}"; do
        local gpu_id=${PREFILL_GPU_ARRAY[$i]}
        local port=${PREFILL_PORT_ARRAY[$i]}
        local kv_port=$((21001 + i))

        echo "  Prefill server $((i+1)): GPU $gpu_id, Port $port, KV Port $kv_port"
        CUDA_VISIBLE_DEVICES=$gpu_id aphrodite run "$MODEL" \
        --host 0.0.0.0 \
        --port "$port" \
        --tensor-parallel-size 1 \
        --seed 1024 \
        --dtype float16 \
        --max-model-len 10000 \
        --max-num-batched-tokens 10000 \
        --max-num-seqs 256 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --kv-transfer-config \
        "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_buffer_size\":\"1e1\",\"kv_port\":\"$kv_port\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"$PROXY_PORT\",\"http_port\":\"$port\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\"}}" > prefill$((i+1)).log 2>&1 &
        PIDS+=($!)
    done

    # =============================================================================
    # Launch Decode Servers (Y Decoders)
    # =============================================================================
    echo ""
    echo "Starting ${#DECODE_GPU_ARRAY[@]} decode server(s)..."
    for i in "${!DECODE_GPU_ARRAY[@]}"; do
        local gpu_id=${DECODE_GPU_ARRAY[$i]}
        local port=${DECODE_PORT_ARRAY[$i]}
        local kv_port=$((22001 + i))

        echo "  Decode server $((i+1)): GPU $gpu_id, Port $port, KV Port $kv_port"
        CUDA_VISIBLE_DEVICES=$gpu_id aphrodite run "$MODEL" \
        --host 0.0.0.0 \
        --port "$port" \
        --tensor-parallel-size 1 \
        --seed 1024 \
        --dtype float16 \
        --max-model-len 10000 \
        --max-num-batched-tokens 10000 \
        --max-num-seqs 256 \
        --trust-remote-code \
        --gpu-memory-utilization 0.7 \
        --kv-transfer-config \
        "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_consumer\",\"kv_buffer_size\":\"8e9\",\"kv_port\":\"$kv_port\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"$PROXY_PORT\",\"http_port\":\"$port\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\"}}" > decode$((i+1)).log 2>&1 &
        PIDS+=($!)
    done

    # =============================================================================
    # Wait for All Servers to Start
    # =============================================================================
    echo ""
    echo "Waiting for all servers to start..."
    for port in "${PREFILL_PORT_ARRAY[@]}" "${DECODE_PORT_ARRAY[@]}"; do
        if ! wait_for_server "$port"; then
            echo "Failed to start server on port $port"
            cleanup
            # shellcheck disable=SC2317 # reached if cleanup returns
            exit 1
        fi
    done

    echo ""
    echo "All servers are up. Waiting for instances to register with proxy..."

    # Wait for proxy to have registered instances
    for i in {1..60}; do
        if curl -s http://localhost:10001/health | grep -q '"status": "ok"'; then
            echo "✓ Proxy has registered instances"
            break
        fi
        if [ "$i" -eq 60 ]; then
            echo "✗ Proxy did not register instances within 60 seconds"
            echo "Proxy health status:"
            curl -s http://localhost:10001/health
            cleanup
            # shellcheck disable=SC2317 # reached if cleanup returns
            exit 1
        fi
        echo "  Waiting for instances to register... ($i/60)"
        sleep 1
    done

    echo ""
    echo "============================================"
    echo "All systems ready!"
    echo "============================================"
    echo "Proxy API:     http://localhost:10001"
    echo "Prefill servers: ${PREFILL_PORTS}"
    echo "Decode servers:  ${DECODE_PORTS}"
    echo ""
    
    if [ "$RUN_BENCHMARK" = true ]; then
        # =============================================================================
        # Run Benchmark
        # =============================================================================
        echo "Starting benchmark..."
        cd ../../../benchmarks/
        aphrodite bench run --port 10001 --seed "$(date +%s)" \
            --model "$MODEL" \
            --dataset-name random --random-input-len 7500 --random-output-len 200 \
            --num-prompts 200 --burstiness 100 --request-rate 2 | tee benchmark.log

        echo "Benchmarking done. Cleaning up..."
        cleanup
    else
        echo "Servers are running. To test:"
        echo ""
        echo "  curl -X POST http://localhost:10001/v1/completions \\"
        echo "    -H 'Content-Type: application/json' \\"
        echo "    -d '{"
        echo "      \"model\": \"$MODEL\","
        echo "      \"prompt\": \"Once upon a time\","
        echo "      \"max_tokens\": 50"
        echo "    }'"
        echo ""
        echo "To run benchmark: ./p2p_nccl_disagg.sh --benchmark"
        echo "Press Ctrl+C to stop all services..."
        echo "============================================"

        # Keep script running and wait for Ctrl+C
        wait
    fi
}

main
