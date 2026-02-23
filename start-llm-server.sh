#!/bin/bash

MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen3-14B-Q4_K_M.gguf}"
PORT="${PORT:-8080}"
# Native context for Qwen3-14B is 32768. YaRN flags below extend it to 131072.
# Increase to 65536 or 131072 if you have the VRAM headroom.
CONTEXT_SIZE="${CONTEXT_SIZE:-32768}"
# Note: Model stays loaded in memory while server runs
# Stop the server when not needed to free VRAM

# Auto-detect GPU layers based on available VRAM
auto_detect_gpu_layers() {
    # NVIDIA GPU via nvidia-smi
    if command -v nvidia-smi &> /dev/null; then
        local free_vram
        free_vram=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
        if [ -n "$free_vram" ] && [ "$free_vram" -gt 0 ]; then
            # Reserve 2.5GB for KV cache, compute buffers, and safety margin
            # Each layer ~150MB for Qwen3-14B Q4_K_M
            local reserved=2560
            local usable_vram=$((free_vram - reserved))
            local layers=$((usable_vram / 150))
            [ "$layers" -lt 0 ] && layers=0
            [ "$layers" -gt 64 ] && layers=64
            echo "$layers"
            return
        fi
    fi

    # AMD GPU via rocm-smi
    if command -v rocm-smi &> /dev/null; then
        if rocm-smi --showmeminfo vram 2>/dev/null | grep -q "Total Memory"; then
            echo "999"  # Let llama.cpp use all available VRAM
            return
        fi
    fi

    # Vulkan GPU (AMD APU, Intel, etc.) — detected via llama-server --version
    if command -v llama-server &> /dev/null; then
        if llama-server --version 2>&1 | grep -q "ggml_vulkan: Found [^0]"; then
            echo "999"  # UMA/Vulkan: GPU can access all system RAM
            return
        fi
    fi

    echo "0"  # No GPU detected — CPU only
}

# Use provided GPU_LAYERS or auto-detect
if [ -z "$GPU_LAYERS" ]; then
    GPU_LAYERS=$(auto_detect_gpu_layers)
    echo "Auto-detected GPU layers: $GPU_LAYERS (based on available VRAM)"
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Download it with:"
    echo "  hf download unsloth/Qwen3-32B-GGUF --include 'Qwen3-32B-Q4_K_M.gguf' --local-dir ~/models"
    exit 1
fi

echo "Starting llama.cpp server..."
echo "  Model: $MODEL_PATH"
echo "  Port: $PORT"
echo "  GPU Layers: $GPU_LAYERS"
echo "  Context Size: $CONTEXT_SIZE"
echo ""
echo "Note: Stop server (Ctrl+C) when not in use to free VRAM"
echo "Note: YaRN enabled — context can be extended to 131072 via CONTEXT_SIZE env var"
echo ""

llama-server \
    --model "$MODEL_PATH" \
    --port "$PORT" \
    --n-gpu-layers "$GPU_LAYERS" \
    --ctx-size "$CONTEXT_SIZE" \
    --rope-scaling yarn \
    --rope-scale 4 \
    --yarn-orig-ctx 32768 \
    --flash-attn on \
    --threads 12 \
    --host 0.0.0.0
