#!/bin/bash
# Start LLM server with a small model for testing
# Uses Qwen2.5-0.5B-Instruct which is fast and fits easily in VRAM

set -e

MODEL_DIR="${HOME}/models"
TEST_MODEL="Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
MODEL_PATH="${MODEL_DIR}/${TEST_MODEL}"
PORT="${PORT:-8080}"

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Test model not found. Downloading Qwen2.5-0.5B-Instruct..."
    echo ""
    
    # Create models directory if needed
    mkdir -p "$MODEL_DIR"
    
    # Download from Hugging Face
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF \
            --include "${TEST_MODEL}" \
            --local-dir "$MODEL_DIR"
    else
        echo "huggingface-cli not found. Installing..."
        pip install huggingface_hub
        huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF \
            --include "${TEST_MODEL}" \
            --local-dir "$MODEL_DIR"
    fi
    
    echo ""
    echo "Download complete!"
fi

echo "Starting test LLM server..."
echo "Model: ${TEST_MODEL}"
echo "Port: ${PORT}"
echo ""

# Start llama-server with small model
# 0.5B model easily fits entirely in VRAM
exec llama-server \
    --model "$MODEL_PATH" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --n-gpu-layers 99 \
    --ctx-size 4096 \
    --threads 8
