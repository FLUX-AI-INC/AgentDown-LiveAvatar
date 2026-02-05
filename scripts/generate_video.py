#!/usr/bin/env python3
"""
Agent Down - LiveAvatar Video Generation

Generates talking head videos using LiveAvatar's 4-step distilled model.
Supports both single-GPU (batch) and multi-GPU (real-time 45 FPS) modes.

Multi-GPU Mode (5 GPUs):
  - Achieves 45 FPS real-time streaming
  - Requires 5x H100/H800 80GB GPUs
  - Uses TPP (Timestep-forcing Pipeline Parallelism)
  
Single-GPU Mode:
  - Batch/offline generation
  - Works on single 80GB GPU with FP8
  - ~0.5 FPS (good for pre-rendered content)
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


def check_gpu() -> Tuple[str, float, int]:
    """Check GPU availability and return device info"""
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available")
        return "cpu", 0.0, 0
    
    gpu_count = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"🖥️  GPU: {gpu_name}")
    print(f"💾 VRAM: {vram:.1f} GB per GPU")
    print(f"🔢 GPU Count: {gpu_count}")
    
    if gpu_count >= 5:
        print("✅ Multi-GPU mode available (5+ GPUs detected)")
        print("   Expected performance: ~45 FPS real-time")
    else:
        print("ℹ️  Single-GPU mode (batch generation)")
        print("   For real-time (45 FPS), use 5 GPUs")
    
    return "cuda", vram, gpu_count


def is_multi_gpu_mode() -> bool:
    """Check if we should use multi-GPU real-time mode"""
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    # Multi-GPU mode requires 5 GPUs and can be disabled via env var
    return gpu_count >= 5 and os.environ.get("LIVEAVATAR_SINGLE_GPU", "").lower() != "true"


class LiveAvatarGenerator:
    """
    Video generator using LiveAvatar.
    
    Supports two modes:
    - Single-GPU: Batch generation for pre-rendered content
    - Multi-GPU (5 GPUs): Real-time 45 FPS streaming with TPP parallelism
    
    Runs LiveAvatar's inference script via subprocess for proper integration.
    """
    
    def __init__(self, force_single_gpu: bool = False):
        self.device, self.vram, self.gpu_count = check_gpu()
        self.liveavatar_dir = PROJECT_ROOT / "liveavatar_lib"
        self.model_path = Path(settings.liveavatar.model_path)
        self.base_model_path = Path(settings.liveavatar.base_model_path)
        
        # Determine mode
        self.multi_gpu = self.gpu_count >= 5 and not force_single_gpu
        if self.multi_gpu:
            print("🚀 Using MULTI-GPU mode (5 GPUs, 45 FPS real-time)")
        else:
            print("📦 Using SINGLE-GPU mode (batch generation)")
        
        # Generation parameters
        self.save_fps = 25  # LiveAvatar default
        
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
    
    def generate_multi_gpu(
        self,
        reference_image: Path,
        audio_path: Path,
        output_path: Path,
        prompt: str = "",
        num_inference_steps: int = 4,
        resolution: str = "720p",
        num_clips: int = 10000,  # Effectively infinite for streaming
        enable_compile: bool = True,
        **kwargs
    ) -> Path:
        """
        Generate video using multi-GPU real-time mode (45 FPS).
        
        Requires 5 GPUs with 80GB VRAM each.
        Uses TPP (Timestep-forcing Pipeline Parallelism).
        """
        self.load_model()
        
        # Resolution for multi-GPU (can use higher res)
        if resolution == '720p':
            size = "720*400"  # Optimized for multi-GPU
        elif resolution == '480p':
            size = "704*384"
        else:
            size = "720*400"
        
        print(f"🚀 Generating video with LiveAvatar MULTI-GPU (45 FPS)...")
        print(f"   Reference: {reference_image}")
        print(f"   Audio: {audio_path}")
        print(f"   Resolution: {size}")
        print(f"   Steps: {num_inference_steps}")
        print(f"   GPUs: 5 (4 DiT + 1 VAE)")
        print(f"   Compile: {enable_compile}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Environment for multi-GPU
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.liveavatar_dir) + ":" + env.get("PYTHONPATH", "")
        env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"
        env["NCCL_DEBUG"] = "WARN"
        env["NCCL_DEBUG_SUBSYS"] = "OFF"
        env["ENABLE_COMPILE"] = "true" if enable_compile else "false"
        
        # Multi-GPU command using torchrun with 5 processes
        cmd = [
            "torchrun",
            "--nproc_per_node=5",
            "--master_port=29102",
            str(self.liveavatar_dir / "minimal_inference" / "s2v_streaming_interact.py"),
            "--ulysses_size", "1",
            "--task", "s2v-14B",
            "--size", size,
            "--ckpt_dir", str(self.base_model_path),  # Base model checkpoint directory
            "--base_seed", "420",
            "--training_config", str(self.liveavatar_dir / "liveavatar" / "configs" / "s2v_causal_sft.yaml"),
            "--offload_model", "False",  # Keep on GPU for speed
            "--convert_model_dtype",
            "--prompt", prompt or "A person speaking naturally, professional studio lighting",
            "--image", str(reference_image.absolute()),
            "--audio", str(audio_path.absolute()),
            "--infer_frames", "48",
            "--load_lora",
            "--lora_path_dmd", str(self.model_path / "liveavatar.safetensors"),  # Must be file, not directory
            "--sample_steps", str(num_inference_steps),
            "--sample_guide_scale", "0",
            "--num_clip", str(num_clips),
            "--num_gpus_dit", "4",  # 4 GPUs for DiT
            "--sample_solver", "euler",
            "--enable_vae_parallel",  # 1 GPU for VAE
            "--save_dir", str(output_path.parent),
        ]
        
        # FP8 only works on H100/H800 (Hopper architecture), not A100
        gpu_name = torch.cuda.get_device_name(0).lower()
        if "h100" in gpu_name or "h800" in gpu_name:
            cmd.append("--fp8")
            print("   FP8: Enabled (H100/H800 detected)")
        else:
            print(f"   FP8: Disabled ({gpu_name} - not Hopper architecture)")
        
        print(f"   Running: torchrun --nproc_per_node=5 s2v_streaming_interact.py...")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.liveavatar_dir),
                env=env,
                capture_output=True,
                text=True,
            )
            
            if result.stdout:
                print("=== STDOUT (last 3000 chars) ===")
                print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
            if result.stderr:
                print("=== STDERR (last 3000 chars) ===")
                print(result.stderr[-3000:] if len(result.stderr) > 3000 else result.stderr)
            
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
            
            # Find output video
            return self._find_output_video(output_path)
            
        except subprocess.CalledProcessError as e:
            print(f"❌ LiveAvatar multi-GPU failed with return code {e.returncode}")
            raise
    
    def _find_output_video(self, output_path: Path) -> Path:
        """Find and move the output video to the expected location"""
        import shutil
        
        search_dirs = [
            output_path.parent,
            output_path.parent.parent,
        ]
        
        mp4_files = []
        for search_dir in search_dirs:
            if search_dir.exists():
                mp4_files.extend(list(search_dir.glob("*.mp4")))
                mp4_files.extend(list(search_dir.glob("video*.mp4")))
        
        if mp4_files:
            mp4_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            source_video = mp4_files[0]
            shutil.move(str(source_video), str(output_path))
            print(f"✅ Video saved: {output_path}")
            return output_path
        else:
            print(f"Looking for videos in: {search_dirs}")
            for d in search_dirs:
                if d.exists():
                    print(f"  Contents of {d}: {list(d.iterdir())}")
            raise FileNotFoundError(f"No output video found in {search_dirs}")
    
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
        
        # Calculate num_clips based on audio duration if not explicitly set high
        # Each clip is ~48 frames at 25fps = ~1.92 seconds
        # We need to cover the entire audio
        try:
            import subprocess as sp
            result = sp.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
                 "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
                capture_output=True, text=True
            )
            audio_duration = float(result.stdout.strip())
            clip_duration = 48 / 25  # ~1.92 seconds per clip
            calculated_clips = int(audio_duration / clip_duration) + 1
            num_clips = max(num_clips, calculated_clips)
            print(f"🎬 Audio duration: {audio_duration:.1f}s, need {num_clips} clips")
        except Exception as e:
            print(f"⚠️ Could not calculate audio duration: {e}, using {num_clips} clips")
        
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
        
        # Use smaller resolution for single GPU to fit in memory
        # 704*384 is recommended for single GPU in LiveAvatar docs
        single_gpu_size = "704*384"
        
        # Use s2v_streaming_interact.py - simpler than batch_eval.py for single samples
        cmd = [
            "python",
            str(self.liveavatar_dir / "minimal_inference" / "s2v_streaming_interact.py"),
            "--task", "s2v-14B",
            "--size", single_gpu_size,  # Use smaller size for single GPU
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
            "--offload_kv_cache",  # Offload KV cache to save VRAM
            "--fp8",  # Enable FP8 quantization to reduce memory
        ]
        
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
            return self._find_output_video(output_path)
                    
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
        
        # Use multi-GPU if available, otherwise single-GPU
        if self.multi_gpu:
            return self.generate_multi_gpu(
                reference_image=reference_image,
                audio_path=audio_path,
                output_path=output_path,
                prompt=prompt,
                **kwargs
            )
        else:
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
