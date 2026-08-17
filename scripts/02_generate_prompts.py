"""
Step 2: Generate Cosmos Transfer prompts from variation ontology using GPT-4 Mini.

Usage:
    python scripts/02_generate_prompts.py \
        --input path/to/ontology.json \
        --target "White surface" \
        --variations "change color to blue" "change material to marble" \
        --output path/to/prompts.json

    # Or generate manually without GPT-4:
    python scripts/02_generate_prompts.py \
        --input path/to/ontology.json \
        --manual \
        --prompt "The video captures a person cleaning a solid blue countertop..."
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_ontology
from src.prompt_generator import generate_prompts_batch


def main():
    parser = argparse.ArgumentParser(description="Generate Cosmos Transfer prompts")
    parser.add_argument("--input", "-i", required=True, help="Path to ontology JSON")
    parser.add_argument("--target", "-t", required=True, help="Target object name")
    parser.add_argument("--variations", "-v", nargs="+", help="Variation descriptions")
    parser.add_argument("--output", "-o", required=True, help="Output prompts JSON")
    parser.add_argument("--manual", action="store_true", help="Use manual prompt instead of GPT-4")
    parser.add_argument("--prompt", help="Manual prompt string (use with --manual)")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=500, help="Max tokens per generated prompt")
    args = parser.parse_args()

    ontology = load_ontology(args.input)

    if args.manual:
        if not args.prompt:
            parser.error("--prompt is required when using --manual")
        results = [{"variation": "manual", "prompt": args.prompt}]
    else:
        if not args.variations:
            parser.error("--variations is required when not using --manual")
        results = generate_prompts_batch(
            ontology, args.target, args.variations, model=args.model,
            temperature=args.temperature, max_tokens=args.max_tokens,
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Saved {len(results)} prompts to {args.output}")


if __name__ == "__main__":
    main()
