#!/usr/bin/env python3
"""
Agent Down - LiveAvatar Video Generation

Generates talking head videos using LiveAvatar's 4-step distilled model.
Supports real-time streaming and infinite length generation.
"""
from __future__ import annotations

import os
import sys
import json
import time
import math
from pathlib import Path
from typing import Optional, Dict, Tuple

import torch
import numpy as np
import PIL.Image

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings, CHARACTERS, ASSETS_DIR, OUTPUT_DIR, MODELS_DIR


def check_gpu() -> Tuple[str, torch.dtype]:
    """Check GPU availability and return device/dtype"""
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU (will be slow)")
        return "cpu", torch.float32
    
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"🖥️  GPU: {gpu_name}")
    print(f"💾 VRAM: {vram:.1f} GB")
    
    # LiveAvatar works best with bfloat16
    dtype = torch.bfloat16
    print("   Using bfloat16")
    
    return device, dtype


class LiveAvatarGenerator:
    """
    Video generator using LiveAvatar.
    
    Supports:
    - 4-step distilled inference (real-time capable)
    - Infinite length streaming
    - FP8 quantization for lower VRAM
    """
    
    def __init__(self, model_path: Optional[str] = None, base_model_path: Optional[str] = None):
        self.device, self.dtype = check_gpu()
        self.model_path = Path(model_path or settings.liveavatar.model_path)
        self.base_model_path = Path(base_model_path or settings.liveavatar.base_model_path)
        self.pipe = None
        
        # Generation parameters (LiveAvatar defaults)
        self.save_fps = 16
        self.num_frames = 93
        self.num_cond_frames = 13
        
    def load_model(self):
        """Load the LiveAvatar pipeline"""
        print(f"📦 Loading LiveAvatar...")
        print(f"   Model path: {self.model_path}")
        print(f"   Base model path: {self.base_model_path}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"LiveAvatar model not found at {self.model_path}\n"
                f"Download with: huggingface-cli download Quark-Vision/Live-Avatar --local-dir {self.model_path}"
            )
        
        if not self.base_model_path.exists():
            raise FileNotFoundError(
                f"Wan base model not found at {self.base_model_path}\n"
                f"Download with: huggingface-cli download Wan-AI/Wan2.2-S2V-14B --local-dir {self.base_model_path}"
            )
        
        # Import LiveAvatar components
        try:
            from liveavatar.pipeline import LiveAvatarPipeline
            from liveavatar.utils import load_audio_features
        except ImportError:
            # Fallback: try to import from cloned repo
            liveavatar_path = PROJECT_ROOT / "liveavatar_lib"
            if liveavatar_path.exists():
                sys.path.insert(0, str(liveavatar_path))
                from liveavatar.pipeline import LiveAvatarPipeline
                from liveavatar.utils import load_audio_features
            else:
                raise ImportError(
                    "LiveAvatar library not found. Please clone it:\n"
                    "git clone https://github.com/Alibaba-Quark/LiveAvatar.git liveavatar_lib"
                )
        
        # Load pipeline
        print("   Loading pipeline...")
        self.pipe = LiveAvatarPipeline.from_pretrained(
            self.base_model_path,
            lora_path=self.model_path / "liveavatar.safetensors",
            torch_dtype=self.dtype,
        )
        self.pipe.to(self.device)
        
        # Enable optimizations
        if settings.liveavatar.enable_compile:
            print("   Compiling model (first run will be slow)...")
            self.pipe.enable_compile()
        
        if settings.liveavatar.enable_fp8:
            print("   Enabling FP8 quantization...")
            self.pipe.enable_fp8()
        
        print("✅ LiveAvatar pipeline loaded")
    
    def generate_single(
        self,
        reference_image: Path,
        audio_path: Path,
        output_path: Path,
        prompt: str = "",
        num_inference_steps: int = 4,
        resolution: str = "480p",
        num_segments: int = 1,
        **kwargs
    ) -> Path:
        """Generate video from reference image and audio"""
        
        if self.pipe is None:
            self.load_model()
        
        # Load reference image
        from diffusers.utils import load_image
        image = load_image(str(reference_image))
        
        # Parse resolution
        if resolution == '480p':
            height, width = 480, 832
        elif resolution == '720p':
            height, width = 768, 1280
        else:
            height, width = 480, 832
        
        print(f"🎬 Generating video...")
        print(f"   Reference: {reference_image}")
        print(f"   Audio: {audio_path}")
        print(f"   Resolution: {width}x{height}")
        print(f"   Steps: {num_inference_steps}")
        
        # Prepare audio
        import librosa
        speech_array, sr = librosa.load(str(audio_path), sr=16000)
        
        # Calculate duration and segments
        audio_duration = len(speech_array) / sr
        generate_duration = self.num_frames / self.save_fps + (num_segments - 1) * (self.num_frames - self.num_cond_frames) / self.save_fps
        
        # Pad audio if needed
        added_samples = math.ceil((generate_duration - audio_duration) * sr)
        if added_samples > 0:
            speech_array = np.append(speech_array, [0.0] * added_samples)
        
        # Generate
        negative_prompt = "Close-up, Bright tones, overexposed, static, blurred details, subtitles, ugly, deformed"
        
        generator = torch.Generator(device=self.device).manual_seed(42)
        
        output = self.pipe(
            image=image,
            audio=speech_array,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=self.num_frames,
            num_inference_steps=num_inference_steps,
            generator=generator,
            num_segments=num_segments,
        )
        
        # Save video
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        from liveavatar.utils import save_video_ffmpeg
        save_video_ffmpeg(
            output.frames,
            str(output_path.with_suffix('')),
            str(audio_path),
            fps=self.save_fps,
            quality=5
        )
        
        print(f"✅ Video saved: {output_path}")
        return output_path
    
    def generate_for_host(
        self,
        host: str,
        audio_path: Path,
        output_path: Path,
        **kwargs
    ) -> Path:
        """Generate video for a specific Agent Down host"""
        
        if host not in CHARACTERS:
            raise ValueError(f"Unknown host: {host}. Available: {list(CHARACTERS.keys())}")
        
        character = CHARACTERS[host]
        
        # Map host names to folder names
        folder_map = {
            "SIGMA-7": "sigma",
            "NOVA": "nova",
            "GLITCH": "glitch",
        }
        folder_name = folder_map.get(host, host.lower())
        
        # Find avatar image
        reference_image = ASSETS_DIR / folder_name / "avatar.png"
        
        if not reference_image.exists():
            alt_paths = [
                ASSETS_DIR / folder_name / "reference.png",
                ASSETS_DIR / host.lower() / "avatar.png",
            ]
            for alt in alt_paths:
                if alt.exists():
                    reference_image = alt
                    break
        
        if not reference_image.exists():
            raise FileNotFoundError(f"Avatar not found for {host} at {reference_image}")
        
        # Build prompt from character
        prompt = f"{character.visual_description}, speaking naturally, professional studio lighting"
        
        return self.generate_single(
            reference_image=reference_image,
            audio_path=audio_path,
            output_path=output_path,
            prompt=prompt,
            **kwargs
        )


# Convenience function
def generate_video(
    reference_image: Path,
    audio_path: Path,
    output_path: Path,
    **kwargs
) -> Path:
    """Generate video using LiveAvatar"""
    generator = LiveAvatarGenerator()
    return generator.generate_single(
        reference_image=reference_image,
        audio_path=audio_path,
        output_path=output_path,
        **kwargs
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate video with LiveAvatar")
    parser.add_argument("--image", type=str, required=True, help="Reference image path")
    parser.add_argument("--audio", type=str, required=True, help="Audio file path")
    parser.add_argument("--output", type=str, required=True, help="Output video path")
    parser.add_argument("--prompt", type=str, default="", help="Text prompt")
    parser.add_argument("--steps", type=int, default=4, help="Inference steps")
    parser.add_argument("--resolution", type=str, default="480p", choices=["480p", "720p"])
    
    args = parser.parse_args()
    
    generate_video(
        reference_image=Path(args.image),
        audio_path=Path(args.audio),
        output_path=Path(args.output),
        prompt=args.prompt,
        num_inference_steps=args.steps,
        resolution=args.resolution,
    )
