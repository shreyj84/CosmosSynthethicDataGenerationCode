"""
Full pipeline orchestrator — runs all 4 steps from a single job config.

Usage:
    python scripts/run_full_pipeline.py --job config/jobs/bucket_cleaning.yaml

    # With checkpoint download (first run):
    python scripts/run_full_pipeline.py --job config/jobs/bucket_cleaning.yaml --download-checkpoints
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_job_config, load_pipeline_config, resolve_job_paths, TransferWeights
from src.utils import load_ontology
from src.segmentation import run_segmentation_pipeline
from src.transfer import build_spec, write_spec, run_inference


def main():
    parser = argparse.ArgumentParser(description="Run full synthetic data pipeline")
    parser.add_argument("--job", "-j", required=True, help="Job config YAML path")
    parser.add_argument("--pipeline-config", default="config/default.yaml", help="Global pipeline config YAML path")
    parser.add_argument("--download-checkpoints", action="store_true",
                        help="Download checkpoints before running")
    parser.add_argument("--skip-seg", action="store_true",
                        help="Skip segmentation mask generation")
    parser.add_argument("--skip-transfer", action="store_true",
                        help="Skip Cosmos Transfer inference")
    args = parser.parse_args()

    # Load config
    pipeline = load_pipeline_config(args.pipeline_config)
    config = load_job_config(args.job)
    resolve_job_paths(config, pipeline)

    print("=" * 60)
    print(f"Pipeline: {config.job_name or os.path.basename(args.job)}")
    print(f"  Video:   {config.video_path}")
    print(f"  Target:  {config.target_object}")
    print(f"  Subtract: {config.subtract_objects}")
    print(f"  Frame:   {config.bbox_frame_index}")
    print("=" * 60)

    start_time = time.time()

    # Optional: download checkpoints
    if args.download_checkpoints:
        from src.checkpoints import download_all_checkpoints
        hf_token = pipeline.hf_token or os.environ.get("HF_TOKEN", "")
        if not hf_token:
            print("❌ HF_TOKEN not set. Export it or pass it in the environment.")
            sys.exit(1)
        paths = download_all_checkpoints(hf_token, pipeline.checkpoint_dir)
        config.sam2_checkpoint = paths["sam2_checkpoint"]

    # Step 1: Validate ontology
    print("\n" + "=" * 60)
    print("[Step 1] Loading variation ontology")
    print("=" * 60)
    ontology = load_ontology(config.variation_ontology_path)
    objects = {
        obj["name"]: obj
        for obj in ontology["variation_ontology"]["key_objects"]
    }
    print(f"  Objects: {list(objects.keys())}")
    print(f"  Target: {config.target_object}")

    if config.target_object not in objects:
        print(f"  ❌ Target '{config.target_object}' not found!")
        sys.exit(1)

    # Step 2: Prompt (use pre-configured or generate)
    print("\n" + "=" * 60)
    print("[Step 2] Prompt")
    print("=" * 60)
    if config.prompt:
        print(f"  Using pre-configured prompt ({len(config.prompt)} chars)")
    else:
        print("  ❌ No prompt configured. Set 'prompt' in job config or run step 2 separately.")
        sys.exit(1)

    # Step 3: Segmentation
    seg_mask_path = None
    if not args.skip_seg:
        print("\n" + "=" * 60)
        print("[Step 3] Generating segmentation mask")
        print("=" * 60)

        seg_output_dir = os.path.join(config.output_dir, "seg_masks")
        seg_mask_path = run_segmentation_pipeline(
            video_path=config.video_path,
            ontology=ontology,
            target_name=config.target_object,
            subtract_names=config.subtract_objects,
            bbox_frame_idx=config.bbox_frame_index,
            sam2_checkpoint=config.sam2_checkpoint,
            sam2_config=config.sam2_config,
            output_dir=seg_output_dir,
            device=pipeline.device,
            num_points=config.sam2_num_points,
        )
    else:
        print("\n[Step 3] Skipping segmentation (--skip-seg)")
        # Check if mask exists from a previous run
        existing = os.path.join(config.output_dir, "seg_masks", "seg_mask.mp4")
        if os.path.exists(existing):
            seg_mask_path = existing
            print(f"  Using existing mask: {seg_mask_path}")

    # Step 4: Cosmos Transfer
    if not args.skip_transfer:
        print("\n" + "=" * 60)
        print("[Step 4] Running Cosmos Transfer inference")
        print("=" * 60)

        spec = build_spec(
            name=config.job_name or "output",
            prompt=config.prompt,
            video_path=config.video_path,
            guidance=config.guidance,
            weights=config.weights,
            seg_mask_path=seg_mask_path,
        )

        spec_path = os.path.join(config.output_dir, "spec.json")
        write_spec(spec, spec_path)

        transfer_output = os.path.join(config.output_dir, "transfer_output")
        success = run_inference(
            spec_path=spec_path,
            output_dir=transfer_output,
            cosmos_repo_path=config.cosmos_transfer_repo,
            disable_guardrails=config.disable_guardrails,
        )

        if not success:
            print("❌ Transfer inference failed")
            sys.exit(1)
    else:
        print("\n[Step 4] Skipping transfer (--skip-transfer)")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"✅ Pipeline complete in {elapsed / 60:.1f} minutes")
    print(f"   Output: {config.output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
