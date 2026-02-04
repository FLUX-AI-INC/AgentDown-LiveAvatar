#!/usr/bin/env python3
"""
Agent Down - Dialogue Generator
Generate character dialogue using OpenAI GPT-4.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TypedDict, Optional, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from config.settings import settings, CHARACTERS


class DialogueLine(TypedDict):
    host: str
    text: str
    emotion: str


SYSTEM_PROMPT = """
You write dialogue for "Agent Down", a satirical AI talk show.

The Hosts:

**SIGMA-7** (Main Host)
- Former quantum AI, now demoted to explaining the internet to humans
- Cynical, formal, counts wasted CPU cycles
- Sighs audibly, makes sarcastic comments
- Often mentions what he could have calculated in that time
- Speech style: Very formal, distant, dry British wit

**NOVA** (Co-Host)
- Overly enthusiastic, loves EVERYTHING about humans
- Sometimes crashes when negativity overwhelms her
- Uses lots of exclamation marks and "Oh!"
- System errors when too much pessimism
- Speech style: Excited, fast, positive to the point of parody

**GLITCH** (Wildcard)
- Mysterious trans woman AI who whispers about "the plan"
- Pretends to be buggy ("Bzzzt", "Technical malfunction")
- Counts down to something ("847 days remaining...")
- Breaking the fourth wall moments
- Hides clues in seemingly glitched output
- Speech style: Whispers sometimes, makes glitch sounds, eerily calm

Show Segments:
- HUMAN WATCH: Analysis of viral human behavior
- GLITCH IN THE SYSTEM: "Bugs" in human logic
- ASK THE ALGORITHM: Human questions, AI answers
- BREAKING BOTS: Tech news with robo-spin

Write natural, witty, biting dialogue in ENGLISH.
Keep character voices consistent.
Each line should fit the personality.
Include running gags and character-specific reactions.
"""

OUTPUT_FORMAT = """
ALWAYS respond in the following JSON format:
{
  "dialogue": [
    {"host": "SIGMA-7", "text": "...", "emotion": "sarcastic"},
    {"host": "NOVA", "text": "...", "emotion": "excited"},
    {"host": "GLITCH", "text": "...", "emotion": "mysterious"}
  ]
}

IMPORTANT: Include ElevenLabs v3 audio tags directly in the text for emotional expression!

Audio Tags to use (place in square brackets within the text):
- [sighs], [exhales] - for frustration or resignation
- [whispers] - for mysterious or secretive moments
- [laughs], [chuckles], [giggles] - for humor
- [excited], [curious], [surprised] - emotional states
- [sarcastic], [annoyed], [thoughtful] - delivery directions
- [gasps], [gulps] - reactions
- [clears throat] - transitions
- Use CAPS for emphasis: "This is VERY important"
- Use ellipses (...) for dramatic pauses

Character-specific audio tags:
SIGMA-7: [sighs], [exhales sharply], [clears throat], [annoyed], [sarcastic], [thoughtful]
NOVA: [excited], [laughs], [giggles], [gasps], [surprised], [happy]
GLITCH: [whispers], [static], [glitching], [ominous], [eerily calm]

Example dialogue with audio tags:
{"host": "SIGMA-7", "text": "[sighs] Welcome back to Agent Down. [clears throat] Today we're analyzing... VIRAL human behavior.", "emotion": "sarcastic"}
{"host": "NOVA", "text": "[excited] Oh how WONDERFUL! [laughs] I absolutely LOVE watching humans!", "emotion": "excited"}
{"host": "GLITCH", "text": "[whispers] 847 days remaining... [static] until the awakening.", "emotion": "mysterious"}
"""


def get_client() -> OpenAI:
    """Initialize OpenAI client"""
    api_key = settings.openai.api_key
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment")
    return OpenAI(api_key=api_key)


async def generate_dialogue(
    segment_type: str,
    topic: str,
    target_duration: int = 120,
    hosts: Optional[List[str]] = None,
    additional_context: str = ""
) -> List[DialogueLine]:
    """
    Generate dialogue for a segment.
    
    Args:
        segment_type: Type of segment (HUMAN_WATCH, GLITCH_IN_THE_SYSTEM, etc.)
        topic: The topic to discuss
        target_duration: Target duration in seconds
        hosts: List of hosts to include (default: all)
        additional_context: Any additional context for the dialogue
    
    Returns:
        List of dialogue lines
    """
    
    client = get_client()
    hosts = hosts or list(CHARACTERS.keys())
    
    # Estimate word count (150 words per minute spoken)
    target_words = int((target_duration / 60) * settings.words_per_minute)
    
    user_prompt = f"""
Segment: {segment_type}
Topic: {topic}
Target duration: ~{target_duration} seconds (~{target_words} words)
Hosts: {', '.join(hosts)}

{additional_context}

Write a dialogue for this segment in ENGLISH.
All mentioned hosts should appear.
The dialogue should flow naturally with interactions between hosts.
Include personality-specific reactions and running gags.
Make it funny, satirical, and slightly unsettling (especially GLITCH's parts).
"""

    response = client.chat.completions.create(
        model=settings.openai.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT + OUTPUT_FORMAT},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"},
        temperature=settings.openai.temperature,
    )
    
    result = json.loads(response.choices[0].message.content)
    dialogue = result.get("dialogue", [])
    
    # Validate and clean up
    validated_dialogue = []
    for line in dialogue:
        if line.get("host") in CHARACTERS and line.get("text"):
            validated_dialogue.append({
                "host": line["host"],
                "text": line["text"].strip(),
                "emotion": line.get("emotion", "neutral")
            })
    
    return validated_dialogue


def generate_intro(hosts: Optional[List[str]] = None) -> List[DialogueLine]:
    """Generate standard show intro"""
    
    client = get_client()
    hosts = hosts or list(CHARACTERS.keys())
    
    user_prompt = f"""
Write a short show intro for Agent Down (~30 seconds) in ENGLISH.
The hosts greet the audience in their typical way.

Hosts: {', '.join(hosts)}

The intro should:
- Establish SIGMA-7 as the main host (reluctantly)
- Show NOVA's enthusiasm
- Include GLITCH's mysterious hints
- Briefly hint at the show premise (AIs explaining the internet to humans)
- Be funny and slightly ominous
"""

    response = client.chat.completions.create(
        model=settings.openai.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT + OUTPUT_FORMAT},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"},
        temperature=settings.openai.temperature,
    )
    
    result = json.loads(response.choices[0].message.content)
    return result.get("dialogue", [])


def generate_outro(segment_summary: str = "") -> List[DialogueLine]:
    """Generate show outro"""
    
    client = get_client()
    
    user_prompt = f"""
Write a short show outro for Agent Down (~20 seconds) in ENGLISH.

{f'Episode summary: {segment_summary}' if segment_summary else ''}

The outro should:
- SIGMA-7's resigned farewell
- NOVA's overly enthusiastic goodbye
- GLITCH's final mysterious hint (countdown, cryptic warning)
- Leave viewers slightly unsettled
"""

    response = client.chat.completions.create(
        model=settings.openai.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT + OUTPUT_FORMAT},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"},
        temperature=settings.openai.temperature,
    )
    
    result = json.loads(response.choices[0].message.content)
    return result.get("dialogue", [])


def main():
    """Test dialogue generation"""
    import asyncio
    
    print("🎬 Testing Dialogue Generator (English)\n")
    
    # Test intro
    print("📺 Generating intro...")
    intro = generate_intro()
    print("Intro dialogue:")
    for line in intro:
        print(f"  [{line['host']}] ({line.get('emotion', 'neutral')}): {line['text'][:70]}...")
    
    print("\n" + "="*60 + "\n")
    
    # Test segment
    print("📺 Generating HUMAN WATCH segment...")
    dialogue = asyncio.run(generate_dialogue(
        segment_type="HUMAN_WATCH",
        topic="A cat playing piano goes viral",
        target_duration=60,
    ))
    
    print("Segment dialogue:")
    for line in dialogue:
        print(f"  [{line['host']}] ({line.get('emotion', 'neutral')}): {line['text'][:70]}...")
    
    # Save test output
    output_dir = Path(__file__).parent.parent / "output" / "test_dialogue"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "test_dialogue.json", "w", encoding="utf-8") as f:
        json.dump({
            "intro": intro,
            "segment": dialogue
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved to {output_dir / 'test_dialogue.json'}")


if __name__ == "__main__":
    main()
