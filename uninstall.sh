#!/bin/bash
# Uninstall 3AM

INSTALL_DIR="${HOME}/.local/share/llm-unified"
BIN_DIR="${HOME}/.local/bin"
CONFIG_DIR="${HOME}/.config/llm-unified"
SYSTEMD_DIR="${HOME}/.config/systemd/user"

# Colors
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}Uninstalling 3AM...${NC}"

# Stop services
systemctl --user stop llm-unified 2>/dev/null || true
systemctl --user stop llama-server 2>/dev/null || true
systemctl --user disable llm-unified 2>/dev/null || true
systemctl --user disable llama-server 2>/dev/null || true

# Remove systemd services
rm -f "$SYSTEMD_DIR/llm-unified.service"
rm -f "$SYSTEMD_DIR/llama-server.service"
systemctl --user daemon-reload

# Remove launcher
rm -f "$BIN_DIR/llm-server"

echo ""
echo -e "${YELLOW}Remove application files? (keeps user data)${NC}"
read -p "Remove $INSTALL_DIR/*.py and static/? [y/N] " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -f "$INSTALL_DIR"/*.py
    rm -rf "$INSTALL_DIR/static"
    rm -rf "$INSTALL_DIR/torque_clustering"
    rm -rf "$INSTALL_DIR/venv"
    echo "Application files removed."
fi

echo ""
echo -e "${RED}Remove ALL user data (conversations, memories, settings)?${NC}"
read -p "This cannot be undone! [y/N] " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "$INSTALL_DIR"
    rm -rf "$CONFIG_DIR"
    echo "All data removed."
else
    echo "User data preserved in $INSTALL_DIR"
fi

echo ""
echo "Uninstall complete."
