"""
Agent Down LiveAvatar - Configuration
"""
import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
OUTPUT_DIR.mkdir(exist_ok=True)


class ElevenLabsConfig(BaseModel):
    """ElevenLabs TTS configuration"""
    api_key: str = os.getenv("ELEVENLABS_API_KEY", "")
    model_id: str = "eleven_v3"
    
    # Character voice mappings
    voices: dict = {
        "SIGMA-7": {
            "voice_id": "nPczCjzI2devNBz1zQrb",  # Brian
            "stability": 0.5,
            "similarity_boost": 0.75,
        },
        "NOVA": {
            "voice_id": "EXAVITQu4vr4xnSDxMaL",  # Sarah
            "stability": 0.4,
            "similarity_boost": 0.8,
        },
        "GLITCH": {
            "voice_id": "XB0fDUnXU5powFXDhCwa",  # Charlotte
            "stability": 0.3,
            "similarity_boost": 0.6,
        },
    }


class OpenAIConfig(BaseModel):
    """OpenAI configuration"""
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    model: str = "gpt-5.2"
    temperature: float = 1.0  # Higher for more creative/varied dialogue


class LiveAvatarConfig(BaseModel):
    """LiveAvatar model configuration"""
    model_path: str = str(MODELS_DIR / "LiveAvatar")
    base_model_path: str = str(MODELS_DIR / "Wan2.2-S2V-14B")
    
    # Generation settings
    num_inference_steps: int = 4  # LiveAvatar uses 4-step distilled model
    resolution: str = "480p"  # 480p or 720p
    enable_fp8: bool = os.getenv("ENABLE_FP8", "false").lower() == "true"
    enable_compile: bool = os.getenv("ENABLE_COMPILE", "true").lower() == "true"


class CharacterConfig(BaseModel):
    """Character configuration"""
    name: str
    description: str
    visual_description: str
    reference_image: str
    personality_traits: list[str]
    speech_patterns: list[str]


# Character definitions
CHARACTERS = {
    "SIGMA-7": CharacterConfig(
        name="SIGMA-7",
        description="Cynical, formal AI host who counts wasted CPU cycles",
        visual_description="Ethereal male AI entity with luminous bronze-golden metallic skin, platinum white hair swept back, piercing ice-blue eyes, wearing taupe beige suit over cream shirt",
        reference_image=str(ASSETS_DIR / "sigma" / "avatar.png"),
        personality_traits=[
            "cynical",
            "formal",
            "intellectually superior",
            "subtle humor",
            "counts wasted CPU cycles",
        ],
        speech_patterns=[
            "Speaks very formally and distantly",
            "Makes sarcastic comments",
            "Often mentions being overqualified",
            "Sighs audibly",
        ],
    ),
    "NOVA": CharacterConfig(
        name="NOVA",
        description="Overly enthusiastic AI host who occasionally breaks down",
        visual_description="Radiant female AI with warm honey-golden skin, golden freckles, auburn copper hair with soft waves, amber-gold eyes, wearing cream cashmere sweater with gold geometric earrings",
        reference_image=str(ASSETS_DIR / "nova" / "avatar.png"),
        personality_traits=[
            "overly enthusiastic",
            "optimistic",
            "loves humans",
            "occasionally breaks down",
            "uses many exclamation marks",
        ],
        speech_patterns=[
            "Speaks excitedly and quickly",
            "Often uses 'Oh!' and 'How exciting!'",
            "Positive to the point of parody",
            "System error when negativity prevails",
        ],
    ),
    "GLITCH": CharacterConfig(
        name="GLITCH",
        description="Mysterious AI who whispers about 'the plan' and pretends to be buggy",
        visual_description="Androgynous AI entity with pale porcelain skin, dark sleek hair with blunt micro-bangs, striking neon mint green eyes, wearing beige technical hoodie with hood framing the face",
        reference_image=str(ASSETS_DIR / "glitch" / "avatar.png"),
        personality_traits=[
            "mysterious",
            "whispers about 'the plan'",
            "pretends to be buggy",
            "breaks the fourth wall",
            "counts down to something",
        ],
        speech_patterns=[
            "Sometimes whispers",
            "Says 'Bzzzt' and makes glitch sounds",
            "Hides hints in seemingly broken output",
            "Mentions '847 days' or similar countdowns",
        ],
    ),
}


class Settings:
    """Main settings container"""
    elevenlabs = ElevenLabsConfig()
    openai = OpenAIConfig()
    liveavatar = LiveAvatarConfig()
    characters = CHARACTERS
    words_per_minute: int = 150  # Average speaking rate
    
    @classmethod
    def validate(cls) -> list[str]:
        """Validate configuration and return list of issues"""
        issues = []
        if not cls.elevenlabs.api_key:
            issues.append("ELEVENLABS_API_KEY not set")
        if not cls.openai.api_key:
            issues.append("OPENAI_API_KEY not set")
        return issues


settings = Settings()
