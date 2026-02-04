#!/usr/bin/env python3
"""
Agent Down - LiveAvatar Video Generation

Generates talking head videos using LiveAvatar's 4-step distilled model.
Uses subprocess to run LiveAvatar's inference script properly.
"""
from __future__ import annotations

import os
import sys
import json
import time
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Tuple

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings, CHARACTERS, ASSETS_DIR, OUTPUT_DIR, MODELS_DIR


def check_gpu() -> Tuple[str, float]:
    """Check GPU availability and return device info"""
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available")
        return "cpu", 0.0
    
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"🖥️  GPU: {gpu_name}")
    print(f"💾 VRAM: {vram:.1f} GB")
    
    return "cuda", vram


class LiveAvatarGenerator:
    """
    Video generator using LiveAvatar.
    
    Runs LiveAvatar's inference script via subprocess for proper integration.
    """
    
    def __init__(self):
        self.device, self.vram = check_gpu()
        self.liveavatar_dir = PROJECT_ROOT / "liveavatar_lib"
        self.model_path = Path(settings.liveavatar.model_path)
        self.base_model_path = Path(settings.liveavatar.base_model_path)
        
        # Generation parameters
        self.save_fps = 16
        
    def load_model(self):
        """Verify model paths exist"""
        print(f"📦 Verifying LiveAvatar setup...")
        print(f"   LiveAvatar dir: {self.liveavatar_dir}")
        print(f"   Model path: {self.model_path}")
        print(f"   Base model path: {self.base_model_path}")
        
        if not self.liveavatar_dir.exists():
            raise FileNotFoundError(
                f"LiveAvatar library not found at {self.liveavatar_dir}\n"
                f"Clone with: git clone https://github.com/Alibaba-Quark/LiveAvatar.git {self.liveavatar_dir}"
            )
        
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
        
        print("✅ LiveAvatar setup verified")
    
    def generate_single(
        self,
        reference_image: Path,
        audio_path: Path,
        output_path: Path,
        prompt: str = "",
        num_inference_steps: int = 4,
        resolution: str = "480p",
        num_clips: int = 1,
        **kwargs
    ) -> Path:
        """Generate video from reference image and audio using LiveAvatar CLI"""
        
        self.load_model()
        
        # Prepare size parameter
        if resolution == '480p':
            size = "832*480"
        elif resolution == '720p':
            size = "1280*768"
        else:
            size = "832*480"
        
        print(f"🎬 Generating video with LiveAvatar...")
        print(f"   Reference: {reference_image}")
        print(f"   Audio: {audio_path}")
        print(f"   Resolution: {size}")
        print(f"   Steps: {num_inference_steps}")
        print(f"   Clips: {num_clips}")
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Set required environment variables for single-GPU distributed mode
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.liveavatar_dir) + ":" + env.get("PYTHONPATH", "")
        env["MASTER_ADDR"] = "localhost"
        env["MASTER_PORT"] = "29500"
        env["RANK"] = "0"
        env["WORLD_SIZE"] = "1"
        env["LOCAL_RANK"] = "0"
        
        # Use s2v_streaming_interact.py - simpler than batch_eval.py for single samples
        cmd = [
            "python",
            str(self.liveavatar_dir / "minimal_inference" / "s2v_streaming_interact.py"),
            "--task", "s2v-14B",
            "--size", size,
            "--ckpt_dir", str(self.base_model_path),
            "--training_config", str(self.liveavatar_dir / "liveavatar" / "configs" / "s2v_causal_sft.yaml"),
            "--load_lora",
            "--lora_path_dmd", str(self.model_path / "liveavatar.safetensors"),
            "--image", str(reference_image.absolute()),
            "--audio", str(audio_path.absolute()),
            "--prompt", prompt or "A person speaking naturally, professional studio lighting",
            "--save_dir", str(output_path.parent),
            "--sample_steps", str(num_inference_steps),
            "--sample_guide_scale", "0",
            "--sample_solver", "euler",
            "--num_clip", str(num_clips),
            "--infer_frames", "48",
            "--num_gpus_dit", "1",
            "--single_gpu",
            "--offload_model", "True",
            "--convert_model_dtype",
        ]
        
        # Add FP8 if enabled (for memory savings)
        if settings.liveavatar.enable_fp8:
            cmd.append("--fp8")
        
        print(f"   Running: python batch_eval.py (with distributed env vars)...")
        print(f"   Command: {' '.join(cmd)}")
        
        try:
            # Run with output captured so we can see errors
            result = subprocess.run(
                cmd,
                cwd=str(self.liveavatar_dir),
                env=env,
                capture_output=True,
                text=True,
            )
            
            # Print stdout/stderr regardless of success
            if result.stdout:
                print("=== STDOUT ===")
                print(result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout)
            if result.stderr:
                print("=== STDERR ===")
                print(result.stderr[-5000:] if len(result.stderr) > 5000 else result.stderr)
            
            # Check return code
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
            
            # Find output video
            expected_output = output_path.parent / "agent_down.mp4"
            if expected_output.exists():
                # Rename to desired output path
                expected_output.rename(output_path)
                print(f"✅ Video saved: {output_path}")
                return output_path
            else:
                # Look for any mp4 in output dir
                mp4_files = list(output_path.parent.glob("*.mp4"))
                if mp4_files:
                    mp4_files[0].rename(output_path)
                    print(f"✅ Video saved: {output_path}")
                    return output_path
                else:
                    raise FileNotFoundError(f"No output video found in {output_path.parent}")
                    
        except subprocess.CalledProcessError as e:
            print(f"❌ LiveAvatar failed with return code {e.returncode}")
            if hasattr(e, 'stdout') and e.stdout:
                print("STDOUT:", e.stdout[-2000:] if len(e.stdout) > 2000 else e.stdout)
            if hasattr(e, 'stderr') and e.stderr:
                print("STDERR:", e.stderr[-2000:] if len(e.stderr) > 2000 else e.stderr)
            raise
    
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
        prompt = f"{character.visual_description}, speaking naturally, talking, professional studio lighting"
        
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
    parser.add_argument("--clips", type=int, default=1, help="Number of clips")
    
    args = parser.parse_args()
    
    generate_video(
        reference_image=Path(args.image),
        audio_path=Path(args.audio),
        output_path=Path(args.output),
        prompt=args.prompt,
        num_inference_steps=args.steps,
        resolution=args.resolution,
        num_clips=args.clips,
    )
