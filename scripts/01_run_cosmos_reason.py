"""
Step 1: Cosmos Reason — generate a variation ontology for a clip, or validate one
that already exists.

Usage:
    # Generate a new ontology for a single clip:
    python scripts/01_run_cosmos_reason.py --generate --clip path/to/clip.mp4 --output-dir path/to/out/

    # Validate/summarize an existing ontology JSON (e.g. one run_batch_prepare.py
    # already produced, or one restored from a previous run):
    python scripts/01_run_cosmos_reason.py --input path/to/out/ontology.json
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_ontology, get_objects_by_name
from src.config import load_pipeline_config


def print_ontology_summary(result: dict, label: str):
    """result must have the shape src/reason.py:run_reason_on_clip() writes to
    ontology.json — frame_dimensions, frame_index, and a nested variation_ontology."""
    objects = get_objects_by_name(result)

    print(f"Ontology: {label}")
    print(f"Frame dimensions: {result.get('frame_dimensions', 'unknown')}")
    print(f"Frame index: {result.get('frame_index', 'unknown')}")
    print(f"\nTask: {result['variation_ontology']['task']['name']}")
    print(f"Description: {result['variation_ontology']['task']['description']}")

    print(f"\nKey objects ({len(objects)}):")
    for name, bbox in objects.items():
        w = bbox["x2"] - bbox["x1"]
        h = bbox["y2"] - bbox["y1"]
        print(f"  • {name}")
        print(f"    bbox: ({bbox['x1']}, {bbox['y1']}) → ({bbox['x2']}, {bbox['y2']})  [{w}×{h} px]")

    print("\nAvailable variations per object:")
    for obj in result["variation_ontology"]["key_objects"]:
        color_vars = obj.get("variations", {}).get("color_and_appearance", [])
        material_vars = obj.get("variations", {}).get("material_and_texture", [])
        print(f"  {obj['name']}:")
        if color_vars:
            print(f"    Colors: {', '.join(color_vars[:4])}")
        if material_vars:
            print(f"    Materials: {', '.join(material_vars[:4])}")


def main():
    parser = argparse.ArgumentParser(description="Generate or validate a Cosmos Reason ontology")
    parser.add_argument("--input", "-i", help="Path to an existing ontology JSON to validate")
    parser.add_argument("--generate", action="store_true", help="Run Cosmos Reason on a clip instead of validating")
    parser.add_argument("--clip", help="Clip video path (required with --generate)")
    parser.add_argument("--output-dir", "-o", help="Where to write ontology.json (required with --generate)")
    parser.add_argument("--model-id", default="nvidia/Cosmos3-Nano", help="Cosmos Reason model id")
    parser.add_argument("--num-objects", type=int, default=10, help="How many objects to ask the model to identify")
    parser.add_argument("--max-new-tokens", type=int, default=8192, help="Generation length cap")
    parser.add_argument("--frame-samples", type=int, default=30, help="Frames sampled when picking the representative frame")
    parser.add_argument("--frame-window", type=int, default=5, help="Sliding-window size for peak-motion detection")
    parser.add_argument("--pipeline-config", default="config/default.yaml", help="Global pipeline config YAML path")
    args = parser.parse_args()

    if args.generate:
        if not args.clip or not args.output_dir:
            parser.error("--generate requires --clip and --output-dir")

        pipeline = load_pipeline_config(args.pipeline_config)
        hf_token = pipeline.hf_token or os.environ.get("HF_TOKEN", "")
        if not hf_token:
            print("❌ HF_TOKEN not set (env var or pipeline config).")
            sys.exit(1)

        from src.reason import load_reasoner, run_reason_on_clip
        processor, model = load_reasoner(args.model_id, hf_token)
        result = run_reason_on_clip(
            args.clip, args.output_dir, model, processor,
            num_objects=args.num_objects,
            max_new_tokens=args.max_new_tokens,
            frame_n_samples=args.frame_samples,
            frame_window_size=args.frame_window,
        )
        print()
        print_ontology_summary(result, args.clip)
        return

    if not args.input:
        parser.error("--input is required unless --generate is used")

    result = load_ontology(args.input)
    print_ontology_summary(result, args.input)


if __name__ == "__main__":
    main()
