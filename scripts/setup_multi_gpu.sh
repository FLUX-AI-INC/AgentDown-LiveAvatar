#!/bin/bash
# =============================================================================
# Agent Down - LiveAvatar Multi-GPU Setup Script
# 
# This script sets up a 5-GPU Runpod environment for 45 FPS real-time
# avatar generation using LiveAvatar with TPP parallelism.
#
# Requirements:
#   - 5x GPUs with 80GB VRAM each (H100/H800)
#   - ~100GB disk space
#   - Ubuntu 22.04 with CUDA 12.4+
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "🚀 Agent Down - LiveAvatar Multi-GPU Setup"
echo "=============================================="

# Check GPU count
GPU_COUNT=$(nvidia-smi -L | wc -l)
echo "🔍 Detected $GPU_COUNT GPUs"

if [ "$GPU_COUNT" -lt 5 ]; then
    echo "⚠️  WARNING: Only $GPU_COUNT GPUs detected. Real-time mode requires 5 GPUs."
    echo "   Continuing with setup (will use single-GPU mode)..."
else
    echo "✅ 5+ GPUs detected - Real-time mode (45 FPS) available!"
fi

# Show GPU info
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Set working directory
WORKDIR=${WORKDIR:-/workspace/AgentDown-LiveAvatar}
cd $WORKDIR
echo "📁 Working directory: $WORKDIR"

# =============================================================================
# 1. System Dependencies
# =============================================================================
echo ""
echo "📦 Installing system dependencies..."
apt-get update && apt-get install -y ffmpeg git-lfs

# =============================================================================
# 2. Python Environment (using system Python, not venv for multi-GPU)
# =============================================================================
echo ""
echo "🐍 Installing Python dependencies..."

# Install PyTorch 2.8+ with CUDA 12.8 support (required for FlashAttention 3)
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

# Install FlashAttention 3 for H100/H800 (Hopper architecture)
echo "⚡ Installing FlashAttention 3..."
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch280 --extra-index-url https://download.pytorch.org/whl/cu128 || {
    echo "⚠️  FlashAttention 3 failed, trying FlashAttention 2..."
    pip install flash-attn==2.8.3 --no-build-isolation
}

# Install project requirements
pip install -r requirements.txt

# =============================================================================
# 3. Clone LiveAvatar Library
# =============================================================================
echo ""
echo "📥 Cloning LiveAvatar library..."

if [ -d "liveavatar_lib" ]; then
    echo "   LiveAvatar already exists, updating..."
    cd liveavatar_lib && git pull && cd ..
else
    git clone https://github.com/Alibaba-Quark/LiveAvatar.git liveavatar_lib
fi

# Install LiveAvatar dependencies
echo "📦 Installing LiveAvatar dependencies..."
pip install -r liveavatar_lib/requirements.txt

# =============================================================================
# 4. Download Models
# =============================================================================
echo ""
echo "📥 Downloading models (this may take 20-30 minutes)..."

mkdir -p models

# Login to HuggingFace if token is set
if [ -n "$HF_TOKEN" ]; then
    echo "🔑 Logging into HuggingFace..."
    huggingface-cli login --token $HF_TOKEN
fi

# Download base model (Wan2.2-S2V-14B) - ~40GB
if [ ! -d "models/Wan2.2-S2V-14B" ] || [ ! -f "models/Wan2.2-S2V-14B/config.json" ]; then
    echo "📥 Downloading Wan2.2-S2V-14B base model (~40GB)..."
    huggingface-cli download Wan-AI/Wan2.2-S2V-14B --local-dir models/Wan2.2-S2V-14B
else
    echo "✅ Wan2.2-S2V-14B already downloaded"
fi

# Download LiveAvatar LoRA (~1GB)
if [ ! -d "models/LiveAvatar" ] || [ ! -f "models/LiveAvatar/liveavatar.safetensors" ]; then
    echo "📥 Downloading LiveAvatar LoRA (~1GB)..."
    huggingface-cli download Quark-Vision/Live-Avatar --local-dir models/LiveAvatar
else
    echo "✅ LiveAvatar LoRA already downloaded"
fi

# =============================================================================
# 5. Create symlinks for LiveAvatar
# =============================================================================
echo ""
echo "🔗 Setting up LiveAvatar symlinks..."

# Create symlink so liveavatar imports work
if [ ! -L "liveavatar" ]; then
    ln -sf liveavatar_lib/liveavatar liveavatar
fi

# =============================================================================
# 6. Configure Environment
# =============================================================================
echo ""
echo "⚙️  Configuring environment for multi-GPU..."

# Set environment variables for optimal multi-GPU performance
cat >> ~/.bashrc << 'EOF'

# LiveAvatar Multi-GPU Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export ENABLE_COMPILE=true
export ENABLE_FP8=true
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=OFF
export PYTHONPATH=/workspace/AgentDown-LiveAvatar/liveavatar_lib:$PYTHONPATH
EOF

# Apply to current session
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export ENABLE_COMPILE=true
export ENABLE_FP8=true
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=OFF
export PYTHONPATH=$WORKDIR/liveavatar_lib:$PYTHONPATH

# =============================================================================
# 7. Verify Setup
# =============================================================================
echo ""
echo "🔍 Verifying setup..."

# Check imports
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU Count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

# Check LiveAvatar imports
python -c "
import sys
sys.path.insert(0, 'liveavatar_lib')
from liveavatar.models.wan.causal_s2v_pipeline_tpp import WanS2V
print('✅ LiveAvatar imports OK')
"

# =============================================================================
# 8. Summary
# =============================================================================
echo ""
echo "=============================================="
echo "✅ Setup Complete!"
echo "=============================================="
echo ""
echo "GPU Configuration:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

if [ "$GPU_COUNT" -ge 5 ]; then
    echo "🚀 MULTI-GPU MODE READY (45 FPS)"
    echo ""
    echo "To run real-time generation:"
    echo "  python run.py segment --type HUMAN_WATCH --topic 'Your topic' --duration 30"
    echo ""
    echo "First run will be slow (compilation). Subsequent runs: ~45 FPS"
else
    echo "📦 SINGLE-GPU MODE (batch generation)"
    echo ""
    echo "To run batch generation:"
    echo "  python run.py segment --type HUMAN_WATCH --topic 'Your topic' --duration 30"
fi

echo ""
echo "Don't forget to set your API keys in .env:"
echo "  cp .env.example .env"
echo "  nano .env"
echo ""
