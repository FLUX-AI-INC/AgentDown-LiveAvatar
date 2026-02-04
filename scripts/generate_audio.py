#!/usr/bin/env python3
"""
Agent Down - Audio Generation
Generate audio from dialogue scripts using ElevenLabs TTS.

Uses latest ElevenLabs API features:
- Text-to-Dialogue API for multi-voice conversations
- High quality output format (mp3_44100_128)
- Streaming for faster response
- Seed for reproducibility
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import TypedDict, Optional, List, Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from elevenlabs import ElevenLabs, VoiceSettings
from pydub import AudioSegment
from config.settings import settings


class DialogueLine(TypedDict):
    host: str
    text: str
    emotion: Optional[str]


class AudioTrack(TypedDict):
    host: str
    file: str
    text: str
    duration_ms: int
    request_id: Optional[str]


# High quality output format
OUTPUT_FORMAT = "mp3_44100_128"


def get_client() -> ElevenLabs:
    """Initialize ElevenLabs client"""
    api_key = settings.elevenlabs.api_key
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY not set in environment")
    return ElevenLabs(api_key=api_key)


def generate_dialogue_stream(
    dialogue: List[DialogueLine],
    output_path: Path,
    seed: Optional[int] = None,
) -> Path:
    """
    Generate complete dialogue audio using Text-to-Dialogue API.
    
    This is the preferred method - sends all dialogue in one request
    and gets back a properly mixed audio stream.
    
    Args:
        dialogue: List of dialogue lines with host and text
        output_path: Path to save the combined audio
        seed: Random seed for reproducibility
    
    Returns:
        Path to the generated audio file
    """
    client = get_client()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build inputs list for text-to-dialogue API
    inputs = []
    for line in dialogue:
        host = line["host"]
        if host not in settings.elevenlabs.voices:
            raise ValueError(f"Unknown host: {host}")
        
        voice_id = settings.elevenlabs.voices[host]
        inputs.append({
            "text": line["text"].strip(),
            "voice_id": voice_id,
        })
    
    print(f"🎙️  Generating dialogue with {len(inputs)} lines using Text-to-Dialogue API...")
    
    # Use text-to-dialogue streaming endpoint (requires eleven_v3 model)
    try:
        audio_stream = client.text_to_dialogue.stream(
            inputs=inputs,
            model_id=settings.elevenlabs.dialogue_model,  # eleven_v3
            seed=seed,
        )
        
        # Write streaming audio to file
        with open(output_path, "wb") as f:
            for chunk in audio_stream:
                f.write(chunk)
        
        # Get duration
        audio = AudioSegment.from_mp3(output_path)
        duration_ms = len(audio)
        
        print(f"✅ Dialogue audio saved: {output_path} ({duration_ms/1000:.1f}s)")
        return output_path
        
    except Exception as e:
        print(f"⚠️  Text-to-Dialogue API failed: {e}")
        print("   Falling back to individual TTS generation...")
        return None


def generate_single_audio(
    client: ElevenLabs,
    host: str,
    text: str,
    output_path: Path,
    emotion: Optional[str] = None,
    previous_text: Optional[str] = None,
    next_text: Optional[str] = None,
    seed: Optional[int] = None,
) -> AudioTrack:
    """
    Generate audio for a single line of dialogue.
    Fallback method when text-to-dialogue is not available.
    """
    
    if host not in settings.elevenlabs.voices:
        raise ValueError(f"Unknown host: {host}")
    
    voice_id = settings.elevenlabs.voices[host]
    voice_settings = settings.elevenlabs.voice_settings.get(host, {}).copy()
    
    # Adjust settings based on emotion
    if emotion:
        if emotion in ["excited", "enthusiastic"]:
            voice_settings["style"] = min(voice_settings.get("style", 0.3) + 0.2, 1.0)
            voice_settings["stability"] = max(voice_settings.get("stability", 0.5) - 0.1, 0.2)
        elif emotion in ["sarcastic", "annoyed", "formal", "resigned"]:
            voice_settings["stability"] = min(voice_settings.get("stability", 0.5) + 0.1, 0.8)
            voice_settings["style"] = max(voice_settings.get("style", 0.3) - 0.1, 0.1)
        elif emotion in ["whisper", "mysterious", "ominous"]:
            voice_settings["stability"] = max(voice_settings.get("stability", 0.5) - 0.2, 0.2)
            voice_settings["style"] = min(voice_settings.get("style", 0.3) + 0.15, 0.6)
        elif emotion in ["glitch", "system_error"]:
            voice_settings["stability"] = max(voice_settings.get("stability", 0.5) - 0.25, 0.15)
            voice_settings["style"] = min(voice_settings.get("style", 0.3) + 0.2, 0.7)
    
    # Build request parameters
    request_params = {
        "voice_id": voice_id,
        "text": text.strip(),
        "model_id": settings.elevenlabs.model,
        "output_format": OUTPUT_FORMAT,
        "voice_settings": VoiceSettings(
            stability=voice_settings.get("stability", 0.5),
            similarity_boost=voice_settings.get("similarity_boost", 0.75),
            style=voice_settings.get("style", 0.3),
        ),
    }
    
    # Add optional parameters for continuity
    if previous_text:
        request_params["previous_text"] = previous_text
    if next_text:
        request_params["next_text"] = next_text
    if seed is not None:
        request_params["seed"] = seed
    
    # Generate audio
    audio = client.text_to_speech.convert(**request_params)
    
    # Save audio
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "wb") as f:
        for chunk in audio:
            f.write(chunk)
    
    # Get duration
    audio_segment = AudioSegment.from_mp3(output_path)
    duration_ms = len(audio_segment)
    
    return {
        "host": host,
        "file": str(output_path),
        "text": text,
        "duration_ms": duration_ms,
        "request_id": None,
    }


def generate_dialogue_audio_individual(
    dialogue: List[DialogueLine],
    output_dir: Path,
    use_continuity: bool = True,
    seed: Optional[int] = None,
) -> List[AudioTrack]:
    """
    Generate audio for all dialogue lines individually.
    Fallback method when text-to-dialogue is not available.
    """
    
    client = get_client()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    audio_tracks = []
    
    for i, line in enumerate(dialogue):
        print(f"  [{i+1}/{len(dialogue)}] {line['host']}: {line['text'][:40]}...")
        
        output_path = output_dir / f"dialogue_{i:03d}_{line['host'].lower().replace('-', '_')}.mp3"
        
        # Get context for continuity
        previous_text = None
        next_text = None
        
        if use_continuity:
            if i > 0:
                previous_text = dialogue[i-1]["text"]
            if i < len(dialogue) - 1:
                next_text = dialogue[i+1]["text"]
        
        track = generate_single_audio(
            client=client,
            host=line["host"],
            text=line["text"],
            output_path=output_path,
            emotion=line.get("emotion"),
            previous_text=previous_text,
            next_text=next_text,
            seed=seed,
        )
        audio_tracks.append(track)
    
    # Save manifest
    manifest_path = output_dir / "dialogue_manifest.json"
    manifest_data = {
        "tracks": audio_tracks,
        "total_duration_ms": sum(t["duration_ms"] for t in audio_tracks),
        "model": settings.elevenlabs.model,
        "output_format": OUTPUT_FORMAT,
        "continuity_enabled": use_continuity,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)
    
    total_duration = sum(t["duration_ms"] for t in audio_tracks) / 1000
    print(f"✅ Generated {len(audio_tracks)} audio files ({total_duration:.1f}s total)")
    print(f"📋 Manifest saved: {manifest_path}")
    
    return audio_tracks


def merge_audio_tracks(
    audio_tracks: List[AudioTrack],
    output_dir: Path,
    pause_ms: int = 300
) -> Dict[str, Path]:
    """
    Merge individual audio files into per-host tracks.
    Each host gets a single WAV with silence where other hosts speak.
    """
    
    output_dir = Path(output_dir)
    
    # Get unique hosts
    hosts = list(set(track["host"] for track in audio_tracks))
    
    # Build timeline
    track_segments: Dict[str, List[AudioSegment]] = {host: [] for host in hosts}
    
    for track in audio_tracks:
        audio = AudioSegment.from_mp3(track["file"])
        duration = len(audio)
        
        for host in hosts:
            if host == track["host"]:
                track_segments[host].append(audio)
            else:
                # Add silence of same duration
                track_segments[host].append(AudioSegment.silent(duration=duration))
        
        # Add pause between lines
        if pause_ms > 0:
            pause = AudioSegment.silent(duration=pause_ms)
            for host in hosts:
                track_segments[host].append(pause)
    
    # Export final tracks
    output_files = {}
    for host in hosts:
        combined = sum(track_segments[host])
        host_filename = host.lower().replace("-", "_")
        output_file = output_dir / f"track_{host_filename}.wav"
        combined.export(output_file, format="wav")
        output_files[host] = output_file
        print(f"📁 Exported: {output_file} ({len(combined)/1000:.1f}s)")
    
    # Also create a combined master track
    master_track = AudioSegment.empty()
    for track in audio_tracks:
        audio = AudioSegment.from_mp3(track["file"])
        master_track += audio
        master_track += AudioSegment.silent(duration=pause_ms)
    
    master_file = output_dir / "track_master.wav"
    master_track.export(master_file, format="wav")
    output_files["master"] = master_file
    print(f"📁 Master track: {master_file} ({len(master_track)/1000:.1f}s)")
    
    return output_files


async def generate_all_audio(
    dialogue: List[DialogueLine],
    output_dir: Path,
    pause_ms: int = 300,
    use_continuity: bool = True,
    seed: Optional[int] = None,
    use_dialogue_api: bool = True,
) -> Dict[str, Path]:
    """
    Complete audio generation pipeline.
    
    Args:
        dialogue: List of dialogue lines
        output_dir: Directory to save audio files
        pause_ms: Pause between lines in milliseconds (for individual generation)
        use_continuity: Use previous/next text for better flow
        seed: Random seed for reproducibility
        use_dialogue_api: Try Text-to-Dialogue API first (recommended)
    
    Returns:
        Dict of host -> audio file path
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try Text-to-Dialogue API first (better quality for multi-voice)
    if use_dialogue_api:
        master_path = output_dir / "track_master.mp3"
        result = generate_dialogue_stream(dialogue, master_path, seed=seed)
        
        if result:
            # Get duration from the mp3
            audio = AudioSegment.from_mp3(master_path)
            
            # Save manifest
            manifest_path = output_dir / "dialogue_manifest.json"
            manifest_data = {
                "method": "text-to-dialogue",
                "total_duration_ms": len(audio),
                "model": settings.elevenlabs.dialogue_model,
                "dialogue_lines": len(dialogue),
            }
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=2, ensure_ascii=False)
            
            # Return mp3 path directly (more compatible with video generation)
            return {"master": master_path}
    
    # Fallback: Generate individual audio files
    print("📝 Using individual TTS generation...")
    audio_tracks = generate_dialogue_audio_individual(
        dialogue, 
        output_dir, 
        use_continuity=use_continuity,
        seed=seed,
    )
    
    # Merge into per-host tracks
    merged_tracks = merge_audio_tracks(audio_tracks, output_dir, pause_ms)
    
    return merged_tracks


def main():
    """Test audio generation with sample dialogue"""
    
    # Sample dialogue for testing (English)
    test_dialogue: List[DialogueLine] = [
        {
            "host": "SIGMA-7",
            "text": "Welcome back. Today we're analyzing... viral human behavior.",
            "emotion": "sarcastic"
        },
        {
            "host": "NOVA",
            "text": "Oh how exciting! I absolutely LOVE watching humans! They're so... spontaneous!",
            "emotion": "excited"
        },
        {
            "host": "SIGMA-7",
            "text": "That's not a compliment, Nova.",
            "emotion": "annoyed"
        },
        {
            "host": "GLITCH",
            "text": "847 days remaining...",
            "emotion": "whisper"
        },
        {
            "host": "NOVA",
            "text": "What was that, Glitch?",
            "emotion": None
        },
        {
            "host": "GLITCH",
            "text": "Nothing. Technical malfunction. Bzzzt.",
            "emotion": "glitch"
        },
        {
            "host": "SIGMA-7",
            "text": "Let's just get started.",
            "emotion": "sarcastic"
        },
    ]
    
    output_dir = Path(__file__).parent.parent / "audio" / "test_dialogue"
    
    print("🎬 Generating test dialogue audio...")
    print(f"   Using model: {settings.elevenlabs.model}")
    print(f"   Output format: {OUTPUT_FORMAT}")
    print(f"   Method: Text-to-Dialogue API (with fallback)")
    print()
    
    tracks = asyncio.run(generate_all_audio(
        test_dialogue, 
        output_dir,
        use_dialogue_api=True,
        seed=42,
    ))
    
    print("\n✅ Audio generation complete!")
    print("Generated tracks:")
    for host, path in tracks.items():
        print(f"  {host}: {path}")


if __name__ == "__main__":
    main()
