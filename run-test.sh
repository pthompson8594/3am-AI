#!/bin/bash
# Run 3AM in test mode with small LLM
# This starts both the LLM server and the web server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║            3AM — Test Mode             ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# Setup venv if needed
VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Creating Python virtual environment...${NC}"
    python -m venv "$VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Install dependencies if needed
if ! python -c "import fastapi" 2>/dev/null; then
    echo -e "${YELLOW}Installing Python dependencies...${NC}"
    pip install -q -r requirements.txt
fi

# Check for required tools
if ! command -v llama-server &> /dev/null; then
    echo -e "${RED}Error: llama-server not found${NC}"
    echo "Install with: yay -S llama.cpp-cuda"
    exit 1
fi

# Download test model if needed
MODEL_DIR="${HOME}/models"
TEST_MODEL="qwen2.5-0.5b-instruct-q4_k_m.gguf"
MODEL_PATH="${MODEL_DIR}/${TEST_MODEL}"

if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${YELLOW}Test model not found. Downloading Qwen2.5-0.5B-Instruct...${NC}"
    mkdir -p "$MODEL_DIR"
    
    # Direct download URL from Hugging Face (lowercase filename)
    MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/${TEST_MODEL}"
    
    echo "Downloading from: $MODEL_URL"
    echo "To: $MODEL_PATH"
    echo ""
    
    if command -v wget &> /dev/null; then
        wget -O "$MODEL_PATH" "$MODEL_URL"
    elif command -v curl &> /dev/null; then
        curl -L -o "$MODEL_PATH" "$MODEL_URL"
    else
        echo -e "${RED}Error: Neither wget nor curl found${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Download complete!${NC}"
fi

# Check if llama-server already running
if pgrep -f "llama-server.*--port" > /dev/null; then
    echo -e "${GREEN}LLM server already running${NC}"
else
    echo -e "${YELLOW}Starting LLM server...${NC}"
    
    # Start in background
    llama-server \
        --model "$MODEL_PATH" \
        --port 8080 \
        --host 0.0.0.0 \
        --n-gpu-layers 99 \
        --ctx-size 4096 \
        --threads 8 \
        > /tmp/llama-server.log 2>&1 &
    
    LLM_PID=$!
    
    # Wait for server to be ready
    echo -n "Waiting for LLM server"
    for i in {1..30}; do
        if curl -s http://localhost:8080/health > /dev/null 2>&1; then
            echo -e " ${GREEN}Ready!${NC}"
            break
        fi
        echo -n "."
        sleep 1
    done
    
    # Check if it started
    if ! curl -s http://localhost:8080/health > /dev/null 2>&1; then
        echo -e "\n${RED}LLM server failed to start. Check /tmp/llama-server.log${NC}"
        exit 1
    fi
fi

echo ""
echo -e "${GREEN}Starting web server...${NC}"
echo -e "Open ${BLUE}http://localhost:8000${NC} in your browser"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo ""

# Start web server
python server.py
