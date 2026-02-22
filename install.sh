#!/bin/bash
# Install 3AM
# Sets up virtual environment, dependencies, and systemd service

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${HOME}/.local/share/3am"
BIN_DIR="${HOME}/.local/bin"
CONFIG_DIR="${HOME}/.config/3am"
VENV_DIR="${INSTALL_DIR}/venv"
SYSTEMD_DIR="${HOME}/.config/systemd/user"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║            3AM — Installer             ║${NC}"
echo -e "${BLUE}║   Self-Calibrating, Feedback-Driven    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# Check dependencies
echo -e "${YELLOW}Checking dependencies...${NC}"

if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python not found${NC}"
    exit 1
fi

LLAMA_SERVER=""
if command -v llama-server &> /dev/null; then
    LLAMA_SERVER="llama-server"
elif [ -f "/usr/bin/llama-server" ]; then
    LLAMA_SERVER="/usr/bin/llama-server"
fi

if [ -z "$LLAMA_SERVER" ]; then
    echo -e "${YELLOW}Warning: llama-server not found${NC}"
    echo ""
    echo "Install llama.cpp for your GPU:"
    echo ""
    echo "  NVIDIA (CUDA):"
    echo "    Arch: yay -S llama.cpp-cuda"
    echo "    Ubuntu: See https://github.com/ggerganov/llama.cpp#cuda"
    echo ""
    echo "  AMD (ROCm):"
    echo "    Arch: yay -S llama.cpp-rocm"
    echo "    Ubuntu: See https://github.com/ggerganov/llama.cpp#hip"
    echo ""
    echo "  AMD APU / Intel / Vulkan (no ROCm):"
    echo "    Arch: yay -S llama.cpp-vulkan"
    echo "    Ubuntu: See https://github.com/ggerganov/llama.cpp#vulkan"
    echo ""
    echo "  CPU only:"
    echo "    Arch: pacman -S llama.cpp"
    echo ""
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    LLAMA_SERVER="llama-server"  # Assume it will be installed later
else
    echo -e "${GREEN}Found: $LLAMA_SERVER${NC}"
fi

# Create directories
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p "$INSTALL_DIR"
mkdir -p "$BIN_DIR"
mkdir -p "$CONFIG_DIR"
mkdir -p "${INSTALL_DIR}/users"

# Create virtual environment
echo -e "${YELLOW}Setting up Python virtual environment...${NC}"
if [ ! -d "$VENV_DIR" ]; then
    python -m venv "$VENV_DIR"
fi

# Activate and install dependencies
source "$VENV_DIR/bin/activate"
echo -e "${YELLOW}Installing Python dependencies...${NC}"
pip install -q --upgrade pip
pip install -q -r "$SCRIPT_DIR/requirements.txt"

# Copy source files
echo -e "${YELLOW}Installing source files...${NC}"
cp "$SCRIPT_DIR"/*.py "$INSTALL_DIR/"
cp -r "$SCRIPT_DIR/static" "$INSTALL_DIR/"
cp -r "$SCRIPT_DIR/torque_clustering" "$INSTALL_DIR/"
cp "$SCRIPT_DIR/requirements.txt" "$INSTALL_DIR/"

# Create launcher script
echo -e "${YELLOW}Creating launcher...${NC}"
cat > "$BIN_DIR/3am" << EOF
#!/bin/bash
# 3AM Server Launcher

INSTALL_DIR="${INSTALL_DIR}"
VENV_DIR="${VENV_DIR}"

# Activate virtual environment
source "\$VENV_DIR/bin/activate"

# Change to install directory
cd "\$INSTALL_DIR"

# Start server
exec python server.py "\$@"
EOF
chmod +x "$BIN_DIR/3am"

# Create config file if not exists
if [ ! -f "$CONFIG_DIR/config.json" ]; then
    echo -e "${YELLOW}Creating default config...${NC}"
    cat > "$CONFIG_DIR/config.json" << EOF
{
    "llm_server_url": "http://localhost:8080",
    "llm_model": "qwen3-14b",
    "web_port": 8000,
    "introspection_schedule": "03:00",
    "introspection_check_interval": 3600,
    "allow_registration": true,
    "session_timeout_hours": 24,
    "clustering_adjustment_factor": 0.5,
    "gemini_api_key": ""
}
EOF
fi

# Create systemd user service
echo -e "${YELLOW}Creating systemd service...${NC}"
mkdir -p "$SYSTEMD_DIR"
cat > "$SYSTEMD_DIR/3am.service" << EOF
[Unit]
Description=3AM Web Server
After=network.target

[Service]
Type=simple
ExecStart=${BIN_DIR}/3am
Restart=on-failure
RestartSec=5
Environment=LLM_URL=http://localhost:8080

[Install]
WantedBy=default.target
EOF

# Download production model if needed
PROD_MODEL="Qwen3-14B-Q4_K_M.gguf"
PROD_MODEL_PATH="${HOME}/models/${PROD_MODEL}"

if [ ! -f "$PROD_MODEL_PATH" ]; then
    echo -e "${YELLOW}Production model not found.${NC}"
    read -p "Download Qwen3-14B Q4_K_M (~9GB, fits in 10GB VRAM)? [Y/n] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        mkdir -p "${HOME}/models"
        MODEL_URL="https://huggingface.co/Qwen/Qwen3-14B-GGUF/resolve/main/${PROD_MODEL}"
        echo "Downloading from: $MODEL_URL"
        echo "This may take a while..."
        
        if command -v wget &> /dev/null; then
            wget --progress=bar:force -O "$PROD_MODEL_PATH" "$MODEL_URL"
        elif command -v curl &> /dev/null; then
            curl -L --progress-bar -o "$PROD_MODEL_PATH" "$MODEL_URL"
        fi
        
        echo -e "${GREEN}Download complete!${NC}"
    fi
fi

# Detect GPU layers for service file
detect_gpu_layers() {
    if command -v nvidia-smi &> /dev/null; then
        local free_vram
        free_vram=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
        if [ -n "$free_vram" ] && [ "$free_vram" -gt 0 ]; then
            echo "nvidia"
            return
        fi
    fi
    if command -v rocm-smi &> /dev/null; then
        if rocm-smi --showmeminfo vram 2>/dev/null | grep -q "Total Memory"; then
            echo "amd"
            return
        fi
    fi
    if [ -n "$LLAMA_SERVER" ] && command -v "$LLAMA_SERVER" &> /dev/null; then
        if "$LLAMA_SERVER" --version 2>&1 | grep -q "ggml_vulkan: Found [^0]"; then
            echo "vulkan"
            return
        fi
    fi
    echo "cpu"
}

GPU_TYPE=$(detect_gpu_layers)
case "$GPU_TYPE" in
    nvidia) GPU_LAYERS=99 ; CTX=16384 ; PARALLEL=4 ;;
    amd)    GPU_LAYERS=999; CTX=8192  ; PARALLEL=2 ;;
    vulkan) GPU_LAYERS=999; CTX=8192  ; PARALLEL=2 ;;
    *)      GPU_LAYERS=0  ; CTX=4096  ; PARALLEL=1 ;;
esac
echo -e "${GREEN}GPU type: ${GPU_TYPE} — using --n-gpu-layers ${GPU_LAYERS}${NC}"

# Create LLM server service
cat > "$SYSTEMD_DIR/llama-server.service" << EOF
[Unit]
Description=Llama.cpp LLM Server
After=network.target

[Service]
Type=simple
ExecStart=${LLAMA_SERVER} --model ${HOME}/models/${PROD_MODEL} --port 8080 --host 0.0.0.0 --n-gpu-layers ${GPU_LAYERS} --ctx-size ${CTX} --parallel ${PARALLEL}
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
EOF

# Reload systemd
systemctl --user daemon-reload 2>/dev/null || true

# Create system-wide service files for server deployment
echo -e "${YELLOW}Creating system service files (for sudo install)...${NC}"
cat > "$SCRIPT_DIR/3am.service" << EOF
[Unit]
Description=3AM Web Server
After=network.target llama-server.service
Wants=llama-server.service

[Service]
Type=simple
User=$USER
WorkingDirectory=${INSTALL_DIR}
ExecStart=${BIN_DIR}/3am
Restart=on-failure
RestartSec=5
Environment=LLM_URL=http://localhost:8080

[Install]
WantedBy=multi-user.target
EOF

cat > "$SCRIPT_DIR/llama-server.service" << EOF
[Unit]
Description=Llama.cpp LLM Server
After=network.target

[Service]
Type=simple
User=$USER
ExecStart=${LLAMA_SERVER} --model ${HOME}/models/${PROD_MODEL} --port 8080 --host 0.0.0.0 --n-gpu-layers ${GPU_LAYERS} --ctx-size ${CTX} --parallel ${PARALLEL}
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

echo ""
echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         Installation Complete!         ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
echo ""
echo -e "Files installed to: ${BLUE}${INSTALL_DIR}${NC}"
echo -e "Config file: ${BLUE}${CONFIG_DIR}/config.json${NC}"
echo ""
echo -e "${YELLOW}Usage (Desktop - user services):${NC}"
echo ""
echo "  Start manually:"
echo "    3am"
echo ""
echo "  Start as user service:"
echo "    systemctl --user enable 3am llama-server"
echo "    systemctl --user start 3am llama-server"
echo ""
echo -e "${YELLOW}Usage (Server - system services):${NC}"
echo ""
echo "  Install system services:"
echo "    sudo cp $SCRIPT_DIR/llama-server.service /etc/systemd/system/"
echo "    sudo cp $SCRIPT_DIR/3am.service /etc/systemd/system/"
echo "    sudo systemctl daemon-reload"
echo "    sudo systemctl enable 3am"
echo "    sudo systemctl start 3am"
echo ""
echo "  View logs:"
echo "    journalctl -u 3am -f"
echo ""
echo -e "${YELLOW}Firewall (if needed):${NC}"
echo "    sudo ufw allow 8000/tcp   # Web UI"
echo "    # OR"
echo "    sudo firewall-cmd --add-port=8000/tcp --permanent"
echo ""
echo -e "Open ${BLUE}http://localhost:8000${NC} (or server IP) in your browser"
echo ""
