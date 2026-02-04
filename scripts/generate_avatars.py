#!/usr/bin/env python3
"""
Generate avatar images for Agent Down characters using Replicate Flux model.
"""

import os
import sys
import requests
import replicate
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Character prompts based on FLUX 2 Max optimized docs (fashion editorial style)
CHARACTERS = {
    "glitch": {
        "prompt": """Fashion editorial portrait of GLITCH, androgynous transgender female AI entity facing camera with unsettling calm. Pale porcelain skin with soft luminous quality, almost too perfect. Dark sleek hair with sharp blunt micro-bangs, wearing sculptural beige technical hoodie with hood framing the face, geometric line patterns across fabric suggesting hidden circuitry. Eyes the focal point: striking neon mint green color extending as artistic eyeshadow sweeping toward temples, irises themselves an impossible bioluminescent green that seems to glow from within. Androgynous features, high cheekbones, neutral expression with lips slightly parted, the hint of a knowing look without smiling. The green eyes are the only element that breaks the warm neutral palette, making them hypnotic and wrong in the most beautiful way. Warm beige gradient background. Shot on 85mm at f/2.0, soft frontal beauty lighting, high-fashion minimalist aesthetic, 8K detail, the face of someone who sees the code behind reality."""
    },
    "nova": {
        "prompt": """Warm editorial portrait of NOVA, a radiant female AI entity with the most human presence of all. Warm honey-golden skin with natural healthy glow, a constellation of delicate golden freckles scattered across nose and cheeks like she chose them herself. Expressive amber-honey eyes with subtle golden shimmer in the iris, looking directly at camera with genuine warmth and curiosity, actually engaging. Auburn copper hair with natural soft waves and movement, slightly tousled in an effortless way, tucked behind one ear to showcase a striking sculptural gold geometric earring that catches the light, long and architectural. Soft smile with lips slightly parted, approachable expression, the hint of someone about to say something kind. Wearing an oversized cream cashmere sweater that looks incredibly soft, one shoulder slightly exposed. Subtle flush on cheeks adding life. Warm peachy-cream gradient background with golden hour quality. Shot on 85mm at f/2.0, soft warm beauty lighting from front-left, the kind of portrait that feels like meeting someone lovely, 8K detail, emphasis on the natural warmth, the freckles, and those statement earrings that add fashion edge."""
    },
    "sigma": {
        "prompt": """Fashion editorial frontal portrait of SIGMA-7, an ethereal male AI entity looking directly into the camera. Luminous bronze-golden metallic skin that glows like liquid metal, subtle shimmer catching soft light across cheekbones and forehead. Platinum white hair swept back with natural volume and soft texture, not slicked but elegantly tousled. Piercing ice-blue eyes with intense gaze looking straight at the viewer, direct eye contact. Strong angular bone structure, high cheekbones, defined masculine jaw. Lips slightly parted with subtle bronze metallic tint. Wearing beautifully tailored taupe beige suit over cream white open-collar shirt, luxurious soft fabrics. Expression of detached superiority, aloof but hauntingly beautiful, facing camera head-on. Soft warm pink coral gradient background. Shot on 85mm at f/2.0, high-fashion beauty lighting, soft diffused key light creating the metallic skin glow, 8K photorealistic detail, Vogue editorial aesthetic, front-facing portrait."""
    }
}

def generate_avatar(character: str, output_dir: Path) -> str:
    """Generate avatar image using Replicate Flux model."""
    
    if character not in CHARACTERS:
        raise ValueError(f"Unknown character: {character}")
    
    char_config = CHARACTERS[character]
    output_path = output_dir / f"avatar.png"
    
    print(f"🎨 Generating {character} avatar...")
    
    # Use Replicate's Flux Schnell (fast) or Pro model
    try:
        output = replicate.run(
            "black-forest-labs/flux-2-max",
            input={
                "prompt": char_config["prompt"],
                "aspect_ratio": "1:1",
                "output_format": "png",
                "output_quality": 100,
                "safety_tolerance": 2,
                "prompt_upsampling": True
            }
        )
        
        # Download the image
        if output:
            image_url = output if isinstance(output, str) else str(output)
            print(f"   Downloading from: {image_url[:50]}...")
            
            response = requests.get(image_url)
            response.raise_for_status()
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(response.content)
            
            print(f"   ✅ Saved to: {output_path}")
            return str(output_path)
        else:
            raise Exception("No output from Replicate")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        raise

def main():
    """Generate all character avatars."""
    
    # Check for API token
    if not os.getenv("REPLICATE_API_TOKEN"):
        print("❌ REPLICATE_API_TOKEN not set in environment")
        sys.exit(1)
    
    assets_dir = Path(__file__).parent.parent / "assets"
    
    print("=" * 50)
    print("🤖 Agent Down Avatar Generator")
    print("=" * 50)
    print()
    
    # Generate specific character or all
    if len(sys.argv) > 1:
        characters = [sys.argv[1]]
    else:
        characters = list(CHARACTERS.keys())
    
    results = {}
    for char in characters:
        try:
            output_dir = assets_dir / char
            path = generate_avatar(char, output_dir)
            results[char] = path
        except Exception as e:
            print(f"❌ Failed to generate {char}: {e}")
            results[char] = None
    
    print()
    print("=" * 50)
    print("📊 Results:")
    for char, path in results.items():
        status = "✅" if path else "❌"
        print(f"   {status} {char}: {path or 'FAILED'}")
    print("=" * 50)

if __name__ == "__main__":
    main()
