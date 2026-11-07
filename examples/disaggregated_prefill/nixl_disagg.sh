#!/bin/bash

# NIXL Disaggregated Prefill/Decode Example
# This uses the standalone NixlConnector (no LMCache needed)

set -e

# Configuration
MODEL="Qwen/Qwen3-0.6B"
PREFILL_PORT=8100
DECODE_PORT=8200
PROXY_PORT=9000

# Get host IP
SONAR_HOST_IP=${SONAR_HOST_IP:-127.0.0.1}

echo "Using host IP: $SONAR_HOST_IP"
echo "Model: $MODEL"

# Kill any existing instances
pkill -f "sonar.*--port $PREFILL_PORT" || true
pkill -f "sonar.*--port $DECODE_PORT" || true
pkill -f "disagg_prefill_proxy_server.py" || true
sleep 2

# Launch prefill instance (producer)
echo "Launching prefill instance on port $PREFILL_PORT..."
CUDA_VISIBLE_DEVICES=0 \
SONAR_NIXL_SIDE_CHANNEL_HOST=$SONAR_HOST_IP \
SONAR_NIXL_SIDE_CHANNEL_PORT=14590 \
python -m sonar.entrypoints.openai.api_server \
    --model $MODEL \
    --port $PREFILL_PORT \
    --max-model-len 1024 \
    --kv-transfer-config '{
        "kv_connector": "NixlConnector",
        "kv_role": "kv_producer",
        "engine_id": "prefill_engine"
    }' > /tmp/prefill.log 2>&1 &

PREFILL_PID=$!
echo "Prefill instance PID: $PREFILL_PID"

# Wait for prefill to start
echo "Waiting for prefill instance to start..."
timeout 120 bash -c 'until curl -s http://localhost:'$PREFILL_PORT'/health > /dev/null; do sleep 1; done' || {
    echo "Prefill instance failed to start!"
    cat /tmp/prefill.log
    exit 1
}
echo "Prefill instance ready!"

# Launch decode instance (consumer)
echo "Launching decode instance on port $DECODE_PORT..."
CUDA_VISIBLE_DEVICES=1 \
SONAR_NIXL_SIDE_CHANNEL_HOST=$SONAR_HOST_IP \
SONAR_NIXL_SIDE_CHANNEL_PORT=14591 \
python -m sonar.entrypoints.openai.api_server \
    --model $MODEL \
    --port $DECODE_PORT \
    --max-model-len 1024 \
    --kv-transfer-config '{
        "kv_connector": "NixlConnector",
        "kv_role": "kv_consumer",
        "engine_id": "decode_engine"
    }' > /tmp/decode.log 2>&1 &

DECODE_PID=$!
echo "Decode instance PID: $DECODE_PID"

# Wait for decode to start
echo "Waiting for decode instance to start..."
timeout 120 bash -c 'until curl -s http://localhost:'$DECODE_PORT'/health > /dev/null; do sleep 1; done' || {
    echo "Decode instance failed to start!"
    cat /tmp/decode.log
    exit 1
}
echo "Decode instance ready!"

# Launch NIXL-aware proxy server
echo "Launching NIXL proxy server on port $PROXY_PORT..."
python3 ../../benchmarks/disagg_benchmarks/nixl_proxy_server.py \
    --prefill-url http://localhost:$PREFILL_PORT/v1/completions \
    --decode-url http://localhost:$DECODE_PORT/v1/completions \
    --prefill-engine-id "prefill_engine" \
    --decode-engine-id "decode_engine" \
    --prefill-nixl-host "$SONAR_HOST_IP" \
    --prefill-nixl-port 14590 \
    --decode-nixl-host "$SONAR_HOST_IP" \
    --decode-nixl-port 14591 \
    --port $PROXY_PORT > /tmp/proxy.log 2>&1 &

PROXY_PID=$!
echo "Proxy server PID: $PROXY_PID"

# Wait for proxy to start
echo "Waiting for proxy server to start..."
timeout 30 bash -c 'until curl -s http://localhost:'$PROXY_PORT'/health > /dev/null 2>&1; do sleep 1; done' || {
    echo "Proxy server started (or no health endpoint)"
}
echo "Proxy server ready!"

echo ""
echo "========================================="
echo "All services are running!"
echo "========================================="
echo "Prefill: http://localhost:$PREFILL_PORT (PID: $PREFILL_PID)"
echo "Decode:  http://localhost:$DECODE_PORT (PID: $DECODE_PID)"
echo "Proxy:   http://localhost:$PROXY_PORT (PID: $PROXY_PID)"
echo ""
echo "Logs:"
echo "  Prefill: /tmp/prefill.log"
echo "  Decode:  /tmp/decode.log"
echo "  Proxy:   /tmp/proxy.log"
echo ""
echo "Test with:"
echo '  curl http://localhost:'$PROXY_PORT'/v1/completions -H "Content-Type: application/json" -d '"'"'{"model":"'$MODEL'","prompt":"Once upon a time","max_tokens":50}'"'"
echo ""
echo "========================================="
echo "Monitoring logs (Press Ctrl+C to stop)..."
echo "========================================="
echo ""

# Function to show colored log output
show_logs() {
    # Mark last seen line for each log
    local prefill_line=0
    local decode_line=0
    local proxy_line=0
    
    while true; do
        # Check for new lines in proxy log
        if [ -f /tmp/proxy.log ]; then
            local new_proxy=$(wc -l < /tmp/proxy.log 2>/dev/null || echo "0")
            if [ "$new_proxy" -gt "$proxy_line" ]; then
                tail -n +$((proxy_line + 1)) /tmp/proxy.log 2>/dev/null | head -n $((new_proxy - proxy_line)) | while read -r line; do
                    echo -e "\033[1;36m[PROXY]\033[0m $line"
                done
                proxy_line=$new_proxy
            fi
        fi
        
        # Check for new lines in prefill log
        if [ -f /tmp/prefill.log ]; then
            local new_prefill=$(wc -l < /tmp/prefill.log 2>/dev/null || echo "0")
            if [ "$new_prefill" -gt "$prefill_line" ]; then
                tail -n +$((prefill_line + 1)) /tmp/prefill.log 2>/dev/null | head -n $((new_prefill - prefill_line)) | \
                    grep -E "(Processing|completions|External|NIXL|ERROR|WARNING)" | while read -r line; do
                    echo -e "\033[1;32m[PREFILL]\033[0m $line"
                done
                prefill_line=$new_prefill
            fi
        fi
        
        # Check for new lines in decode log
        if [ -f /tmp/decode.log ]; then
            local new_decode=$(wc -l < /tmp/decode.log 2>/dev/null || echo "0")
            if [ "$new_decode" -gt "$decode_line" ]; then
                tail -n +$((decode_line + 1)) /tmp/decode.log 2>/dev/null | head -n $((new_decode - decode_line)) | \
                    grep -E "(Processing|completions|External|NIXL|ERROR|WARNING)" | while read -r line; do
                    echo -e "\033[1;33m[DECODE]\033[0m $line"
                done
                decode_line=$new_decode
            fi
        fi
        
        sleep 0.5
    done
}

# Wait for user interrupt
# shellcheck disable=SC2064 # PIDs are known now; expand at trap definition
trap "echo ''; echo 'Stopping services...'; kill $PREFILL_PID $DECODE_PID $PROXY_PID 2>/dev/null; exit 0" INT TERM

# Start showing logs
show_logs

