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


def _concat_videos(clip_paths: list, output_path: Path):
    """Concatenate video clips using ffmpeg"""
    import subprocess
    import tempfile
    
    # Create ffmpeg concat file
    concat_file = output_path.parent / "concat_list.txt"
    with open(concat_file, "w") as f:
        for clip in clip_paths:
            f.write(f"file '{clip}'\n")
    
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy",
        str(output_path),
    ]
    
    print(f"   🎬 Concatenating {len(clip_paths)} clips...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"   ⚠️ ffmpeg concat failed, trying re-encode...")
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c:v", "libx264",
            "-c:a", "aac",
            str(output_path),
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    
    concat_file.unlink(missing_ok=True)
    print(f"   ✅ Final video: {output_path}")


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
    
    from scripts.dialogue_generator import generate_dialogue, generate_intro, generate_outro
    from scripts.generate_audio import generate_all_audio
    from scripts.generate_video import LiveAvatarGenerator
    import json
    
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
    
    dialogue = []
    
    # Generate intro
    if include_intro:
        print("   Generating intro...")
        intro = generate_intro()
        dialogue.extend(intro)
    
    # Generate main segment
    print(f"   Generating {segment_type} segment...")
    segment_dialogue = await generate_dialogue(
        segment_type=segment_type,
        topic=topic,
        target_duration=duration,
    )
    dialogue.extend(segment_dialogue)
    
    # Generate outro
    if include_outro:
        print("   Generating outro...")
        outro = generate_outro(segment_summary=topic)
        dialogue.extend(outro)
    
    # Save dialogue
    dialogue_path = output_dir / "dialogue.json"
    with open(dialogue_path, "w") as f:
        json.dump(dialogue, f, indent=2)
    print(f"   📝 Saved {len(dialogue)} lines to {dialogue_path}")
    
    # Step 2: Generate audio
    print("\n2️⃣  GENERATING AUDIO (ElevenLabs)...")
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
    
    # Check if we have multiple hosts - if so, use individual TTS
    # so we get per-line audio files for per-host video generation
    hosts_in_dialogue = set(line.get("host", "SIGMA-7") for line in dialogue)
    use_individual = len(hosts_in_dialogue) > 1
    
    if use_individual:
        print(f"   🎭 Multiple hosts detected ({', '.join(hosts_in_dialogue)})")
        print(f"   Using individual TTS for per-host video clips...")
    
    audio_tracks = await generate_all_audio(
        dialogue=dialogue,
        output_dir=audio_dir,
        use_dialogue_api=not use_individual,  # Use individual TTS for multi-host
    )
    
    master_audio = audio_tracks.get("master")
    if not master_audio:
        print("❌ Failed to generate audio")
        return None
    
    print(f"   🎵 Master audio: {master_audio}")
    
    # Step 3: Generate video with LiveAvatar
    print("\n3️⃣  GENERATING VIDEO (LiveAvatar)...")
    
    video_dir = output_dir / "video"
    video_dir.mkdir(exist_ok=True)
    
    generator = LiveAvatarGenerator()
    generator.load_model()
    
    # Split audio per dialogue line and generate video per host
    from pydub import AudioSegment
    
    # Check if we have individual audio tracks per line
    individual_audio_dir = output_dir / "audio"
    individual_files = sorted(individual_audio_dir.glob("dialogue_*.mp3"))
    
    if individual_files and len(individual_files) == len(dialogue):
        # We have per-line audio files - generate per-host video clips
        print(f"   📹 Generating {len(dialogue)} video clips (per host)...")
        
        clip_videos = []
        for i, (line, audio_file) in enumerate(zip(dialogue, individual_files)):
            host = line.get("host", "SIGMA-7")
            clip_output = video_dir / f"clip_{i:03d}_{host.lower()}.mp4"
            
            print(f"   [{i+1}/{len(dialogue)}] {host}: {line.get('text', '')[:50]}...")
            
            try:
                generator.generate_for_host(
                    host=host,
                    audio_path=audio_file,
                    output_path=clip_output,
                    resolution=resolution,
                    num_inference_steps=steps,
                )
                clip_videos.append(clip_output)
            except Exception as e:
                print(f"   ⚠️ Failed clip {i}: {e}")
        
        # Concatenate clips into final video
        if clip_videos:
            output_video = video_dir / "segment.mp4"
            _concat_videos(clip_videos, output_video)
        else:
            print("❌ No clips generated")
            return None
    else:
        # Single master audio - generate per-host clips by splitting audio at dialogue boundaries
        # For now, generate one video per unique host using the master audio
        # Each host gets the full master audio (LiveAvatar handles lip-sync)
        hosts_in_dialogue = []
        seen = set()
        for line in dialogue:
            h = line.get("host", "SIGMA-7")
            if h not in seen:
                hosts_in_dialogue.append(h)
                seen.add(h)
        
        if len(hosts_in_dialogue) == 1:
            # Single host - just generate one video
            print(f"   📹 Single host: {hosts_in_dialogue[0]}")
            output_video = video_dir / "segment.mp4"
            generator.generate_for_host(
                host=hosts_in_dialogue[0],
                audio_path=master_audio,
                output_path=output_video,
                resolution=resolution,
                num_inference_steps=steps,
            )
        else:
            # Multiple hosts - generate separate videos, pick primary for now
            # TODO: Split audio per dialogue line for true multi-host switching
            primary_host = hosts_in_dialogue[0]
            print(f"   📹 Multiple hosts detected: {hosts_in_dialogue}")
            print(f"   📹 Using primary host: {primary_host}")
            print(f"   ℹ️  For multi-character switching, use individual TTS mode")
            
            output_video = video_dir / "segment.mp4"
            generator.generate_for_host(
                host=primary_host,
                audio_path=master_audio,
                output_path=output_video,
                resolution=resolution,
                num_inference_steps=steps,
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
    segment_parser.add_argument("--duration", type=int, default=30, help="Target duration in seconds (default: 30 for testing)")
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
