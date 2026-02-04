# Agent Down - LiveAvatar Edition

Real-time AI talk show generation using LiveAvatar for 45 FPS video synthesis.

## Overview

This project integrates:
- **ElevenLabs** - AI voice generation for character dialogue
- **OpenAI GPT** - Dialogue script generation
- **LiveAvatar** - Real-time audio-driven avatar video (45 FPS)
- **OBS** (optional) - Live streaming integration

## Why LiveAvatar?

| Feature | LiveAvatar | LongCat |
|---------|-----------|---------|
| Speed | **45 FPS real-time** | ~16 min/segment |
| Steps | **4-step sampling** | 50 steps |
| Streaming | Native infinite length | Manual stitching |

## Requirements

- Python 3.10+
- CUDA 12.4+
- GPU with 80GB VRAM (or 48GB with FP8)
- ~50GB disk space for models

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/YOUR_USERNAME/AgentDown-LiveAvatar.git
cd AgentDown-LiveAvatar
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Models

```bash
# LiveAvatar models
huggingface-cli download Wan-AI/Wan2.2-S2V-14B --local-dir ./models/Wan2.2-S2V-14B
huggingface-cli download Quark-Vision/Live-Avatar --local-dir ./models/LiveAvatar
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 4. Generate a Segment

```bash
python run.py segment --type HUMAN_WATCH --topic "Apple's new AI features"
```

## Project Structure

```
AgentDown-LiveAvatar/
├── assets/              # Character avatars
│   ├── sigma/
│   ├── nova/
│   └── glitch/
├── config/              # Configuration
├── scripts/             # Generation scripts
│   ├── generate_dialogue.py
│   ├── generate_audio.py
│   └── generate_video.py
├── models/              # Model weights (gitignored)
├── output/              # Generated content
├── run.py               # Main entry point
└── requirements.txt
```

## Characters

- **SIGMA-7** - Ethereal bronze-skinned AI, detached and superior
- **NOVA** - Warm honey-gold AI with genuine enthusiasm  
- **GLITCH** - Pale androgynous AI with mint green eyes, sees too much

## License

MIT
