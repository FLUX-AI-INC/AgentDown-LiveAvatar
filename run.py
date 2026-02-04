#!/usr/bin/env python3
"""
Agent Down - LiveAvatar Edition

Main entry point for generating AI talk show segments using LiveAvatar.
"""
from __future__ import annotations

import os
import sys
import time
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings, CHARACTERS, OUTPUT_DIR


def check_environment():
    """Check environment and print status"""
    print("🔍 Checking environment...")
    
    issues = settings.validate()
    if issues:
        for issue in issues:
            print(f"   ❌ {issue}")
        return False
    
    # Check GPU
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🖥️  GPU: {gpu_name}")
        print(f"💾 VRAM: {vram:.1f} GB")
    else:
        print("⚠️  No GPU detected - will be slow")
    
    # Check models
    liveavatar_path = Path(settings.liveavatar.model_path)
    base_model_path = Path(settings.liveavatar.base_model_path)
    
    liveavatar_exists = liveavatar_path.exists() and (liveavatar_path / "liveavatar.safetensors").exists()
    base_model_exists = base_model_path.exists()
    
    print(f"📦 Models:")
    print(f"   LiveAvatar: {'✅' if liveavatar_exists else '❌'}")
    print(f"   Wan2.2-S2V: {'✅' if base_model_exists else '❌'}")
    
    if not liveavatar_exists or not base_model_exists:
        print("\n⚠️  Missing models. Download with:")
        if not base_model_exists:
            print(f"   huggingface-cli download Wan-AI/Wan2.2-S2V-14B --local-dir {base_model_path}")
        if not liveavatar_exists:
            print(f"   huggingface-cli download Quark-Vision/Live-Avatar --local-dir {liveavatar_path}")
        return False
    
    print("✅ All checks passed!")
    return True


async def generate_segment(
    segment_type: str,
    topic: str,
    duration: int = 60,
    resolution: str = "480p",
    steps: int = 4,
    include_intro: bool = True,
    include_outro: bool = True,
):
    """Generate a complete show segment"""
    
    from scripts.dialogue_generator import DialogueGenerator
    from scripts.generate_audio import generate_all_audio
    from scripts.generate_video import LiveAvatarGenerator
    
    # Create output directory
    timestamp = int(time.time())
    output_dir = OUTPUT_DIR / f"segment_{segment_type.lower()}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"🎬 AGENT DOWN - LiveAvatar Edition")
    print(f"   Segment: {segment_type}")
    print(f"   Topic: {topic}")
    print(f"   Resolution: {resolution}")
    print(f"   Steps: {steps} (LiveAvatar 4-step distilled)")
    print("=" * 60)
    
    # Step 1: Generate dialogue
    print("\n1️⃣  GENERATING DIALOGUE...")
    dialogue_gen = DialogueGenerator()
    dialogue = dialogue_gen.generate_segment(
        segment_type=segment_type,
        topic=topic,
        duration_seconds=duration,
        include_intro=include_intro,
        include_outro=include_outro
    )
    
    # Save dialogue
    dialogue_path = output_dir / "dialogue.json"
    import json
    with open(dialogue_path, "w") as f:
        json.dump([line.model_dump() for line in dialogue], f, indent=2)
    print(f"   📝 Saved {len(dialogue)} lines to {dialogue_path}")
    
    # Step 2: Generate audio
    print("\n2️⃣  GENERATING AUDIO (ElevenLabs)...")
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
    
    audio_tracks = await generate_all_audio(
        dialogue=dialogue,
        output_dir=audio_dir,
    )
    
    master_audio = audio_tracks.get("master")
    if not master_audio:
        print("❌ Failed to generate audio")
        return None
    
    print(f"   🎵 Master audio: {master_audio}")
    
    # Step 3: Generate video with LiveAvatar
    print("\n3️⃣  GENERATING VIDEO (LiveAvatar)...")
    
    # Determine primary host from dialogue
    primary_host = dialogue[0].speaker if dialogue else "SIGMA-7"
    
    video_dir = output_dir / "video"
    video_dir.mkdir(exist_ok=True)
    
    generator = LiveAvatarGenerator()
    generator.load_model()
    
    # Calculate number of segments based on audio duration
    from pydub import AudioSegment
    audio = AudioSegment.from_mp3(master_audio)
    audio_duration = len(audio) / 1000  # seconds
    
    num_segments = max(1, int(audio_duration / 5))  # ~5 seconds per segment
    
    print(f"   📹 Generating {num_segments} video segments...")
    
    output_video = video_dir / "segment.mp4"
    
    generator.generate_for_host(
        host=primary_host,
        audio_path=master_audio,
        output_path=output_video,
        resolution=resolution,
        num_inference_steps=steps,
        num_segments=num_segments,
    )
    
    # Step 4: Summary
    print("\n" + "=" * 60)
    print("✅ SEGMENT GENERATION COMPLETE!")
    print(f"   📁 Output: {output_dir}")
    print(f"   📝 Dialogue: {dialogue_path}")
    print(f"   🎵 Audio: {master_audio}")
    print(f"   🎬 Video: {output_video}")
    print("=" * 60)
    
    return {
        "output_dir": output_dir,
        "dialogue": dialogue_path,
        "audio": master_audio,
        "video": output_video,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Agent Down - LiveAvatar Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py check                                    # Check environment
  python run.py segment --type HUMAN_WATCH --topic "AI" # Generate segment
  python run.py video --image avatar.png --audio voice.mp3 --output video.mp4
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Check command
    check_parser = subparsers.add_parser("check", help="Check environment")
    
    # Segment command
    segment_parser = subparsers.add_parser("segment", help="Generate a show segment")
    segment_parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["HUMAN_WATCH", "GLITCH_IN_THE_SYSTEM", "ASK_THE_ALGORITHM", "BREAKING_BOTS"],
        help="Segment type"
    )
    segment_parser.add_argument("--topic", type=str, required=True, help="Topic to discuss")
    segment_parser.add_argument("--duration", type=int, default=60, help="Target duration in seconds")
    segment_parser.add_argument("--resolution", type=str, default="480p", choices=["480p", "720p"])
    segment_parser.add_argument("--steps", type=int, default=4, help="Inference steps (4 recommended)")
    segment_parser.add_argument("--no-intro", action="store_true", help="Skip intro")
    segment_parser.add_argument("--no-outro", action="store_true", help="Skip outro")
    
    # Video command (direct video generation)
    video_parser = subparsers.add_parser("video", help="Generate video from image + audio")
    video_parser.add_argument("--image", type=str, required=True, help="Reference image")
    video_parser.add_argument("--audio", type=str, required=True, help="Audio file")
    video_parser.add_argument("--output", type=str, required=True, help="Output video path")
    video_parser.add_argument("--prompt", type=str, default="", help="Text prompt")
    video_parser.add_argument("--steps", type=int, default=4, help="Inference steps")
    video_parser.add_argument("--resolution", type=str, default="480p", choices=["480p", "720p"])
    
    args = parser.parse_args()
    
    if args.command == "check":
        success = check_environment()
        sys.exit(0 if success else 1)
    
    elif args.command == "segment":
        # Check environment first
        if not check_environment():
            sys.exit(1)
        
        asyncio.run(generate_segment(
            segment_type=args.type,
            topic=args.topic,
            duration=args.duration,
            resolution=args.resolution,
            steps=args.steps,
            include_intro=not args.no_intro,
            include_outro=not args.no_outro,
        ))
    
    elif args.command == "video":
        from scripts.generate_video import generate_video
        generate_video(
            reference_image=Path(args.image),
            audio_path=Path(args.audio),
            output_path=Path(args.output),
            prompt=args.prompt,
            num_inference_steps=args.steps,
            resolution=args.resolution,
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
