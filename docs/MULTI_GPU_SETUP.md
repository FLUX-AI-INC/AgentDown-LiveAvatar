# LiveAvatar Multi-GPU Setup (45 FPS Real-Time)

This guide explains how to set up a 5-GPU Runpod instance for real-time avatar generation at 45 FPS.

## Requirements

- **5x GPUs** with 80GB VRAM each (H100/H800 recommended)
- **~100GB disk space** for models
- **Python 3.10+**
- **CUDA 12.4+**

## Performance Comparison

| Mode | GPUs | FPS | Latency | Cost/hr (Runpod) |
|------|------|-----|---------|------------------|
| Single-GPU | 1x H100 | ~0.5 | High (batch) | ~$2.39 |
| Multi-GPU | 5x H100 PCIe | **45** | Low (real-time) | ~$6.75 |
| Multi-GPU | 5x H100 SXM | **45** | Low (real-time) | ~$11.95 |

## Runpod Setup

### 1. Create a Multi-GPU Pod

1. Go to [Runpod.io](https://runpod.io)
2. Click "Deploy" → "GPU Pod"
3. Select a template with **5 GPUs**:
   - Search for "5x H100" or "5x A100"
   - Or use "Secure Cloud" and filter for 5+ GPU machines
4. Configuration:
   - **Container Image**: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
   - **Volume Size**: 150GB (for models + outputs)
   - **Volume Mount**: `/workspace`

### 2. Verify GPU Setup

After connecting to your pod:

```bash
# Check GPU count (should show 5 GPUs)
nvidia-smi

# Should show:
# GPU 0: NVIDIA H100 80GB HBM3
# GPU 1: NVIDIA H100 80GB HBM3
# GPU 2: NVIDIA H100 80GB HBM3
# GPU 3: NVIDIA H100 80GB HBM3
# GPU 4: NVIDIA H100 80GB HBM3
```

### 3. Run Setup Script

```bash
cd /workspace
git clone https://github.com/AlexanderCGO2/AgentDown-LiveAvatar.git
cd AgentDown-LiveAvatar
bash scripts/setup_multi_gpu.sh
```

This script will:
- Install dependencies (PyTorch, FlashAttention 3, etc.)
- Clone the LiveAvatar library
- Download models (~50GB)
- Configure environment

### 4. Configure API Keys

```bash
cp .env.example .env
nano .env  # Add your API keys

# Required:
OPENAI_API_KEY=sk-...
ELEVENLABS_API_KEY=...
HF_TOKEN=...  # For model downloads
```

### 5. Test Real-Time Generation

```bash
# Quick test (should achieve 45 FPS)
python run.py segment --type HUMAN_WATCH --topic "Hello World test" --duration 10
```

## Architecture

LiveAvatar uses **Timestep-forcing Pipeline Parallelism (TPP)**:

```
┌─────────────────────────────────────────────────────────────┐
│                    5-GPU TPP Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│  GPU 0-3: DiT Model (4-step denoising in parallel)         │
│    └── Each GPU handles one denoising timestep              │
│    └── Linear speedup: 4x faster than sequential            │
│                                                             │
│  GPU 4: VAE Decoder (parallel video decoding)               │
│    └── Converts latents to video frames                     │
│    └── Runs in parallel with next DiT batch                 │
└─────────────────────────────────────────────────────────────┘
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_COMPILE` | `true` | Enable torch.compile (slow first run, faster after) |
| `ENABLE_FP8` | `true` | Enable FP8 quantization (saves VRAM) |
| `LIVEAVATAR_SINGLE_GPU` | `false` | Force single-GPU mode even with 5 GPUs |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3,4` | Which GPUs to use |

## Troubleshooting

### OOM Errors
- Ensure all 5 GPUs have 80GB VRAM
- Try lowering resolution: `--resolution 480p`
- Disable compilation: `export ENABLE_COMPILE=false`

### Slow First Run
- This is normal! Compilation takes 5-10 minutes on first run
- Subsequent runs will be much faster (45 FPS)

### NCCL Errors
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
# Re-run to see detailed NCCL logs
```

### GPU Not Detected
```bash
# Check CUDA visibility
echo $CUDA_VISIBLE_DEVICES

# Should be: 0,1,2,3,4
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
```

## Cost Estimates

| Usage | GPUs | Hours/Month | Monthly Cost |
|-------|------|-------------|--------------|
| Light testing | 5x H100 PCIe | 10 hrs | ~$68 |
| Regular use | 5x H100 PCIe | 40 hrs | ~$270 |
| Heavy production | 5x H100 PCIe | 160 hrs | ~$1,080 |
| 24/7 streaming | 5x H100 PCIe | 720 hrs | ~$4,860 |

## Next Steps

Once running at 45 FPS, you can:
1. **Stream to OBS**: Use the generated video as a virtual camera
2. **Integrate with LLM**: Connect to GPT-5 for real-time conversation
3. **Add TTS**: Pipe ElevenLabs audio directly to LiveAvatar
