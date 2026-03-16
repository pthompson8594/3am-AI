#!/bin/bash
# Sync RPC worker install files to a remote machine.
#
# Reads per-node config from rpc-workers.json if present.
#
# Usage:
#   ./sync-rpc-worker.sh <user> <host>
#   ./sync-rpc-worker.sh pthompson13 192.168.11.20
#   SSH_KEY=~/.ssh/id_ed25519 ./sync-rpc-worker.sh pthompson13 192.168.11.20

set -e

USER="$1"
HOST="$2"

if [ -z "$USER" ] || [ -z "$HOST" ]; then
    echo "Usage: $0 <user> <host>"
    echo "  e.g. $0 pthompson13 192.168.11.20"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_DIR="/home/${USER}/rpc-worker"
CONFIG_FILE="${SCRIPT_DIR}/rpc-workers.json"

SSH_ARGS="-o StrictHostKeyChecking=accept-new"
if [ -n "$SSH_KEY" ]; then
    SSH_ARGS="$SSH_ARGS -i $SSH_KEY"
fi

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}RPC Worker Sync${NC}"
echo "  To: ${USER}@${HOST}:${REMOTE_DIR}"

# ── Resolve node config from rpc-workers.json ─────────────────────────────────
CONF_TMPDIR=""
NODE_CONF_FILE=""
if [ -f "$CONFIG_FILE" ]; then
    CONF_TMPDIR="$(mktemp -d /tmp/rpc-sync.XXXXXX)"
    NODE_CONF_FILE="${CONF_TMPDIR}/rpc-worker.conf.json"
    trap 'rm -rf "$CONF_TMPDIR"' EXIT

    python3 - <<PYEOF > "$NODE_CONF_FILE"
import json

with open("$CONFIG_FILE") as f:
    cfg = json.load(f)

defaults = cfg.get("default", {})
node     = cfg.get("nodes", {}).get("$HOST", {})
merged   = {**defaults, **node}
merged.pop("_comment", None)

print(json.dumps(merged, indent=2))
PYEOF

    RPC_HOST_CFG=$(python3 -c "import json; d=json.load(open('$NODE_CONF_FILE')); print(d.get('rpc_host','0.0.0.0'))")
    RPC_PORT_CFG=$(python3 -c "import json; d=json.load(open('$NODE_CONF_FILE')); print(d.get('rpc_port',50052))")
    echo "  Config  : rpc_host=${RPC_HOST_CFG}, rpc_port=${RPC_PORT_CFG} (from rpc-workers.json)"
else
    echo -e "  ${YELLOW}rpc-workers.json not found — no node config will be pushed${NC}"
fi

echo ""

# ── Sync files ─────────────────────────────────────────────────────────────────
ssh $SSH_ARGS "${USER}@${HOST}" "mkdir -p ${REMOTE_DIR}"

RSYNC_FILES=(
    "${SCRIPT_DIR}/llama-rpc-worker.service"
    "${SCRIPT_DIR}/install-rpc-worker.sh"
)
if [ -n "$NODE_CONF_FILE" ]; then
    RSYNC_FILES+=("${NODE_CONF_FILE}")
fi

rsync -avzc --progress \
    -e "ssh $SSH_ARGS" \
    "${RSYNC_FILES[@]}" \
    "${USER}@${HOST}:${REMOTE_DIR}/"

ssh $SSH_ARGS "${USER}@${HOST}" "chmod +x ${REMOTE_DIR}/install-rpc-worker.sh"

echo ""
echo -e "${GREEN}Done.${NC} To install on the remote machine:"
echo "  ssh ${USER}@${HOST}"
echo "  cd ~/rpc-worker && sudo ./install-rpc-worker.sh"
