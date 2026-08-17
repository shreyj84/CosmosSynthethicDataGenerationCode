"""
Generate Cosmos Transfer prompts from variation ontology using GPT-4 Mini.

Takes a Cosmos Reason variation ontology JSON and produces detailed text
prompts suitable for Cosmos Transfer's text-guided video generation.
"""

import json
import os
from typing import List, Dict, Any, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


SYSTEM_PROMPT = """You are an expert at writing prompts for NVIDIA Cosmos Transfer 2.5, 
a video-to-video model that transforms input videos based on text descriptions.

Given a scene description and a specific variation to apply, write a detailed prompt 
that describes the ENTIRE scene as it should appear AFTER the transformation.

Rules:
1. Describe the changed object in vivid detail: color, material, texture, finish.
2. Explicitly state that all OTHER objects retain their original appearance.
3. Describe the lighting and camera angle as they are in the original.
4. Repeat the target color/material 2-3 times in different contexts to reinforce it.
5. Keep the prompt under 300 words.
6. Do NOT use phrases like "transform" or "change" — describe the scene as if it 
   already looks the way you want.
7. Be specific about materials: "smooth matte blue finish" not just "blue".

Example output format:
"The video captures a person cleaning a solid blue countertop in a well-lit indoor 
kitchen. The countertop surface is a rich, vivid blue color with a smooth matte 
finish, clearly visible under the even overhead lighting. The person uses paper 
towels and a bucket to wipe down the blue countertop surface. The bucket and paper 
towels retain their original appearance. The room is lit by bright, even overhead 
ceiling lights that illuminate the blue countertop uniformly."
"""


def generate_prompt(
    ontology: Dict[str, Any],
    target_object: str,
    variation_description: str,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 500,
) -> str:
    """
    Generate a Cosmos Transfer prompt using GPT-4 Mini.

    Args:
        ontology: Loaded Cosmos Reason variation ontology dict.
        target_object: Name of the object to change.
        variation_description: What to change about it (e.g. "change color to blue").
        api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
        model: OpenAI model to use.
        temperature: Sampling temperature.
        max_tokens: Max tokens in the generated prompt.

    Returns:
        Generated prompt string.
    """
    if OpenAI is None:
        raise ImportError("openai package not installed. Run: pip install openai")

    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or pass api_key.")

    client = OpenAI(api_key=api_key)

    # Build context from ontology
    location = ontology["variation_ontology"]["location"]
    task = ontology["variation_ontology"]["task"]
    objects = ontology["variation_ontology"]["key_objects"]

    # Find the target object's current appearance
    target_info = None
    other_objects = []
    for obj in objects:
        if obj["name"] == target_object:
            target_info = obj
        else:
            other_objects.append(obj["name"])

    if target_info is None:
        available = [obj["name"] for obj in objects]
        raise KeyError(
            f"Target '{target_object}' not found. Available: {available}"
        )

    user_message = f"""Scene description:
- Room: {location['room_type']}, {location['setting']}
- Lighting: {location['lighting_condition']}
- Task: {task['description']}

Target object to change: {target_object}
- Current appearance: {target_info['current_appearance']['color_description']}
- Current material: {target_info['current_appearance']['material']}
- Current texture: {target_info['current_appearance']['texture']}

Desired variation: {variation_description}

Other objects in scene (must retain original appearance): {', '.join(other_objects)}

Write a Cosmos Transfer prompt describing the scene with this variation applied."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    prompt = response.choices[0].message.content.strip()

    # Strip quotes if the model wrapped it
    if prompt.startswith('"') and prompt.endswith('"'):
        prompt = prompt[1:-1]

    return prompt


def generate_prompts_batch(
    ontology: Dict[str, Any],
    target_object: str,
    variations: List[str],
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 500,
) -> List[Dict[str, str]]:
    """
    Generate multiple prompts for different variations of the same object.

    Args:
        ontology: Loaded Cosmos Reason variation ontology dict.
        target_object: Name of the object to change.
        variations: List of variation descriptions.
        api_key: OpenAI API key.
        model: OpenAI model to use.
        temperature: Sampling temperature.
        max_tokens: Max tokens in each generated prompt.

    Returns:
        List of dicts with 'variation' and 'prompt' keys.
    """
    results = []
    for i, variation in enumerate(variations):
        print(f"  [{i + 1}/{len(variations)}] Generating prompt for: {variation}")
        prompt = generate_prompt(
            ontology, target_object, variation, api_key, model,
            temperature=temperature, max_tokens=max_tokens,
        )
        results.append({"variation": variation, "prompt": prompt})
        print(f"    → {len(prompt)} chars")

    return results


def generate_prompt_manual(
    scene_description: str,
    target_description: str,
    other_objects_description: str,
) -> str:
    """
    Build a Cosmos Transfer prompt manually without GPT-4 Mini.

    This is the fallback when you don't want to use the OpenAI API.

    Args:
        scene_description: What the scene looks like overall.
        target_description: Detailed description of the changed object.
        other_objects_description: What stays the same.

    Returns:
        Formatted prompt string.
    """
    return (
        f"The video captures {scene_description}. "
        f"{target_description}. "
        f"{other_objects_description}. "
        f"The room is lit by bright, even indoor lighting."
    )
