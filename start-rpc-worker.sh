#!/bin/bash
# Run on each remote GPU worker machine to enable distributed inference.
#
# llama.cpp RPC worker — exposes local GPU(s) to the main llama-server.
# The main machine runs llama-server with --rpc <this-host>:$PORT and
# offloads model layers here. No model file needed on worker machines.
#
# Usage:
#   ./start-rpc-worker.sh
#   RPC_PORT=50052 RPC_HOST=0.0.0.0 ./start-rpc-worker.sh

PORT="${RPC_PORT:-50052}"
HOST="${RPC_HOST:-0.0.0.0}"

if ! command -v rpc-server &> /dev/null; then
    echo "Error: rpc-server not found."
    echo ""
    echo "llama.cpp must be built with RPC support:"
    echo "  cmake -B build -DGGML_RPC=ON -DGGML_CUDA=ON   # NVIDIA"
    echo "  cmake -B build -DGGML_RPC=ON -DGGML_HIP=ON    # AMD ROCm"
    echo "  cmake -B build -DGGML_RPC=ON -DGGML_VULKAN=ON # Vulkan"
    echo "  cmake --build build --config Release -j"
    echo ""
    echo "The rpc-server binary will be at build/bin/rpc-server"
    exit 1
fi

# Show the address the main machine should use in --rpc
LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo "Starting llama.cpp RPC worker..."
echo "  Listening: ${HOST}:${PORT}"
if [ -n "$LOCAL_IP" ]; then
    echo "  Main machine connect with: --rpc ${LOCAL_IP}:${PORT}"
fi
echo ""
echo "Keep this running while 3AM is in use. Stop it to remove this"
echo "worker from the inference pool (main server handles it gracefully)."
echo ""

exec rpc-server --host "$HOST" --port "$PORT"
