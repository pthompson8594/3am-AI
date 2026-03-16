#!/bin/bash
# Install llama-rpc-worker as a systemd service on this machine.
#
# Run as root (or with sudo) on each GPU worker node.
#
# Usage:
#   sudo ./install-rpc-worker.sh
#   sudo RPC_PORT=50053 LLAMA_USER=myuser ./install-rpc-worker.sh

set -e

SERVICE_NAME="llama-rpc-worker"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
ENV_FILE="/etc/${SERVICE_NAME}.conf"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Configuration ─────────────────────────────────────────────────────────────
# Load defaults from rpc-worker.conf.json if present (pushed by sync-rpc-worker.sh)
_CONF="${SCRIPT_DIR}/rpc-worker.conf.json"
if [ -f "$_CONF" ] && command -v python3 &>/dev/null; then
    _py() { python3 -c "import json; d=json.load(open('$_CONF')); print(d.get('$1','$2'))"; }
    RPC_HOST="${RPC_HOST:-$(_py rpc_host 0.0.0.0)}"
    RPC_PORT="${RPC_PORT:-$(_py rpc_port 50052)}"
    RPC_THREADS="${RPC_THREADS:-$(_py rpc_threads 12)}"
    RPC_DEVICE="${RPC_DEVICE:-$(_py rpc_device '')}"
    RPC_CACHE="${RPC_CACHE:-$(_py rpc_cache true)}"
    LLAMA_USER="${LLAMA_USER:-$(_py user "$(logname 2>/dev/null || echo llama)")}"
    LLAMA_BIN="${LLAMA_BIN:-$(_py llama_bin '')}"
fi

# Env vars take final precedence; fall back to safe defaults if still unset
RPC_HOST="${RPC_HOST:-0.0.0.0}"
RPC_PORT="${RPC_PORT:-50052}"
RPC_THREADS="${RPC_THREADS:-12}"
RPC_DEVICE="${RPC_DEVICE:-}"
RPC_CACHE="${RPC_CACHE:-true}"
LLAMA_USER="${LLAMA_USER:-$(logname 2>/dev/null || echo llama)}"

# Auto-detect binary if not set by config or env
if [ -z "$LLAMA_BIN" ]; then
    LLAMA_BIN="$(command -v rpc-server 2>/dev/null || echo '')"
fi
# ──────────────────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

if [ "$(id -u)" -ne 0 ]; then
    echo -e "${RED}Error: run this script as root or with sudo.${NC}"
    exit 1
fi

echo -e "${BLUE}llama.cpp RPC Worker — systemd installer${NC}"
echo ""

# Validate binary
if [ -z "$LLAMA_BIN" ]; then
    echo -e "${RED}Error: rpc-server not found on PATH.${NC}"
    echo ""
    echo "Build llama.cpp with RPC support first:"
    echo "  cmake -B build -DGGML_RPC=ON -DGGML_CUDA=ON   # NVIDIA"
    echo "  cmake -B build -DGGML_RPC=ON -DGGML_HIP=ON    # AMD ROCm"
    echo "  cmake -B build -DGGML_RPC=ON -DGGML_VULKAN=ON # Vulkan"
    echo "  cmake --build build --config Release -j"
    echo ""
    echo "Then re-run: LLAMA_BIN=/path/to/rpc-server sudo $0"
    exit 1
fi

echo "  Binary  : $LLAMA_BIN"
echo "  User    : $LLAMA_USER"
echo "  Listen  : ${RPC_HOST}:${RPC_PORT}"
echo "  Threads : ${RPC_THREADS}"
[ -n "$RPC_DEVICE" ] && echo "  Device  : ${RPC_DEVICE}"
[ "$RPC_CACHE" = "true" ] && echo "  Cache   : enabled"
echo ""

# Write environment file (easy to edit later without touching the unit)
cat > "$ENV_FILE" <<EOF
# llama-rpc-worker environment
# Edit this file and run: systemctl restart llama-rpc-worker
RPC_HOST=${RPC_HOST}
RPC_PORT=${RPC_PORT}
RPC_THREADS=${RPC_THREADS}
RPC_DEVICE=${RPC_DEVICE}
RPC_CACHE=$( [ "$RPC_CACHE" = "true" ] && echo 1 || echo '' )
EOF
echo -e "${GREEN}Written${NC} $ENV_FILE"

# Build optional flags resolved at install time (systemd can't do bash conditionals)
EXTRA_FLAGS=""
[ -n "$RPC_DEVICE" ]       && EXTRA_FLAGS="$EXTRA_FLAGS --device ${RPC_DEVICE}"
[ "$RPC_CACHE" = "true" ]  && EXTRA_FLAGS="$EXTRA_FLAGS --cache"

# Write service unit, substituting binary path, user, and optional flags
sed \
    -e "s|^ExecStart=rpc-server|ExecStart=${LLAMA_BIN}|" \
    -e "s|--threads \${RPC_THREADS}|--threads \${RPC_THREADS}${EXTRA_FLAGS}|" \
    -e "s|^User=llama|User=${LLAMA_USER}|" \
    -e "s|^Group=llama|Group=${LLAMA_USER}|" \
    "${SCRIPT_DIR}/llama-rpc-worker.service" > "$SERVICE_FILE"
echo -e "${GREEN}Written${NC} $SERVICE_FILE"

# Reload, enable, and start
systemctl daemon-reload
systemctl enable "${SERVICE_NAME}"
systemctl restart "${SERVICE_NAME}"

echo ""
echo -e "${GREEN}Service started.${NC}"
echo ""

# Show status and the address the main server should use
sleep 1
systemctl status "${SERVICE_NAME}" --no-pager -l || true

echo ""
LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
if [ -n "$LOCAL_IP" ]; then
    echo -e "${BLUE}Add this worker to your main llama-server with:${NC}"
    echo "  --rpc ${LOCAL_IP}:${RPC_PORT}"
fi
echo ""
echo "Useful commands:"
echo "  journalctl -u ${SERVICE_NAME} -f      # live logs"
echo "  systemctl restart ${SERVICE_NAME}      # restart"
echo "  systemctl disable ${SERVICE_NAME}      # remove from boot"
