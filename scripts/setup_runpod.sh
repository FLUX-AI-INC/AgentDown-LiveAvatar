#!/bin/bash
# Agent Down - LiveAvatar Edition
# Runpod Setup Script

set -e

echo "========================================"
echo "🚀 Agent Down - LiveAvatar Setup"
echo "========================================"

# Workspace setup
WORKSPACE=${WORKSPACE:-/workspace}
PROJECT_DIR="$WORKSPACE/AgentDown-LiveAvatar"

# Check if already in project directory
if [[ "$(pwd)" == *"AgentDown-LiveAvatar"* ]]; then
    PROJECT_DIR=$(pwd)
    WORKSPACE=$(dirname "$PROJECT_DIR")
fi

cd "$WORKSPACE"

echo ""
echo "1️⃣  Cloning/Updating repository..."
if [ -d "$PROJECT_DIR" ]; then
    cd "$PROJECT_DIR"
    git pull
else
    git clone https://github.com/AlexanderCGO2/AgentDown-LiveAvatar.git "$PROJECT_DIR"
    cd "$PROJECT_DIR"
fi

echo ""
echo "2️⃣  Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

echo ""
echo "3️⃣  Installing PyTorch with CUDA..."
pip install --upgrade pip
pip install torch==2.8.0 torchvision==0.23.0 torchaudio --index-url https://download.pytorch.org/whl/cu128

echo ""
echo "4️⃣  Installing Flash Attention..."
# For Hopper architecture (H100/H200)
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "H100\|H200"; then
    echo "   Detected Hopper GPU, installing FlashAttention 3..."
    pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch280 --extra-index-url https://download.pytorch.org/whl/cu128 || true
else
    echo "   Installing FlashAttention 2..."
    pip install flash-attn==2.8.3 --no-build-isolation || true
fi

echo ""
echo "5️⃣  Installing project dependencies..."
pip install -r requirements.txt

echo ""
echo "6️⃣  Installing FFmpeg..."
apt-get update && apt-get install -y ffmpeg

echo ""
echo "7️⃣  Cloning LiveAvatar library..."
if [ ! -d "liveavatar_lib" ]; then
    git clone https://github.com/Alibaba-Quark/LiveAvatar.git liveavatar_lib
fi

# Create symlink for imports
if [ ! -L "liveavatar" ]; then
    ln -sf liveavatar_lib/liveavatar liveavatar
fi

echo ""
echo "8️⃣  Downloading models..."
mkdir -p models

# Check for HuggingFace token
if [ -z "$HF_TOKEN" ]; then
    if [ -f ".env" ]; then
        export $(grep HF_TOKEN .env | xargs)
    fi
fi

if [ -n "$HF_TOKEN" ]; then
    huggingface-cli login --token "$HF_TOKEN" || true
fi

# Download Wan base model
if [ ! -d "models/Wan2.2-S2V-14B" ] || [ ! -f "models/Wan2.2-S2V-14B/config.json" ]; then
    echo "   Downloading Wan2.2-S2V-14B (~25GB)..."
    huggingface-cli download Wan-AI/Wan2.2-S2V-14B --local-dir models/Wan2.2-S2V-14B
fi

# Download LiveAvatar LoRA
if [ ! -d "models/LiveAvatar" ] || [ ! -f "models/LiveAvatar/liveavatar.safetensors" ]; then
    echo "   Downloading LiveAvatar LoRA..."
    huggingface-cli download Quark-Vision/Live-Avatar --local-dir models/LiveAvatar
fi

echo ""
echo "========================================"
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Create .env file with API keys:"
echo "     cp .env.example .env"
echo "     # Edit .env with your keys"
echo ""
echo "  2. Run a test:"
echo "     python run.py check"
echo ""
echo "  3. Generate a segment:"
echo "     python run.py segment --type HUMAN_WATCH --topic 'Your topic'"
echo "========================================"
