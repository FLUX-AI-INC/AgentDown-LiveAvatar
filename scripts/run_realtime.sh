#!/bin/bash
# =============================================================================
# Agent Down - Real-Time 45 FPS Generation
# 
# Quick launcher for multi-GPU real-time avatar generation.
# Requires 5x H100/H800 GPUs.
# =============================================================================

set -e

# Verify 5 GPUs
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo "0")
if [ "$GPU_COUNT" -lt 5 ]; then
    echo "❌ Error: Real-time mode requires 5 GPUs. Found: $GPU_COUNT"
    echo ""
    echo "For single-GPU batch mode, use:"
    echo "  python run.py segment --type HUMAN_WATCH --topic 'Your topic'"
    exit 1
fi

# Set multi-GPU environment
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export ENABLE_COMPILE=true
export ENABLE_FP8=true
export NCCL_DEBUG=WARN

# Default values
TOPIC="${1:-Test topic for real-time generation}"
DURATION="${2:-30}"
TYPE="${3:-HUMAN_WATCH}"

echo "🚀 Starting Real-Time Generation (45 FPS)"
echo "   Topic: $TOPIC"
echo "   Duration: ${DURATION}s"
echo "   Segment Type: $TYPE"
echo "   GPUs: 5 (4 DiT + 1 VAE)"
echo ""

# Run generation
cd /workspace/AgentDown-LiveAvatar
python run.py segment \
    --type "$TYPE" \
    --topic "$TOPIC" \
    --duration "$DURATION"
