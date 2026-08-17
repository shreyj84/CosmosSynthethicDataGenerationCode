"""
Phase 2 (interactive, resumable): for each clip scripts/run_batch_prepare.py prepared,
list the objects Cosmos Reason detected on the CLI, let you pick the target object to
change and any objects to protect ("subtract"), pick or edit a generated prompt, then
run SAM2 segmentation and Cosmos Transfer inference for that clip.

Safe to interrupt (Ctrl-C) and re-run — a clip counts as done once its
transfer_output/.done marker exists, so re-running only touches clips that haven't
completed yet.

Usage:
    python scripts/run_interactive_review.py --job config/jobs/bucket_cleaning_batch.yaml
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_batch_job_config, load_pipeline_config, resolve_job_paths
from src.segmentation import run_segmentation_pipeline
from src.transfer import build_spec, write_spec, run_inference


def discover_pending_clips(output_dir):
    """Sorted (clip_name, clip_dir) pairs with an ontology.json but no completed
    transfer_output/.done marker yet."""
    if not os.path.isdir(output_dir):
        return []
    pending = []
    for name in sorted(os.listdir(output_dir)):
        clip_dir = os.path.join(output_dir, name)
        if not os.path.isdir(clip_dir) or name == "_clips":
            continue
        if not os.path.exists(os.path.join(clip_dir, "ontology.json")):
            continue
        if os.path.exists(os.path.join(clip_dir, "transfer_output", ".done")):
            continue
        pending.append((name, clip_dir))
    return pending


def prompt_index(label, count):
    while True:
        raw = input(label).strip()
        try:
            idx = int(raw)
        except ValueError:
            print(f"  Enter a number 1-{count}.")
            continue
        if 1 <= idx <= count:
            return idx - 1
        print(f"  Enter a number 1-{count}.")


def prompt_indices(label, count):
    """Comma-separated indices, or blank for none."""
    while True:
        raw = input(label).strip()
        if not raw:
            return []
        try:
            idxs = [int(x.strip()) for x in raw.split(",") if x.strip()]
        except ValueError:
            print(f"  Enter comma-separated numbers 1-{count}, or leave blank.")
            continue
        if all(1 <= i <= count for i in idxs):
            return [i - 1 for i in idxs]
        print(f"  Enter comma-separated numbers 1-{count}, or leave blank.")


def pick_target_and_subtract(key_objects):
    print("\n  Detected objects:")
    for i, obj in enumerate(key_objects, 1):
        bbox = obj.get("bbox_2d_pixels", {})
        print(f"    {i}. {obj.get('name', '?')} — {obj.get('role_in_task', '')}  "
              f"[{bbox.get('x1')},{bbox.get('y1')},{bbox.get('x2')},{bbox.get('y2')}]")

    target_idx = prompt_index("\n  Pick the object to CHANGE (number): ", len(key_objects))
    target_name = key_objects[target_idx]["name"]

    remaining = [o for i, o in enumerate(key_objects) if i != target_idx]
    if not remaining:
        return target_name, []

    print("\n  Objects to PROTECT — excluded even if they overlap the target (comma-separated numbers, blank for none):")
    for i, obj in enumerate(remaining, 1):
        print(f"    {i}. {obj.get('name', '?')}")
    subtract_idxs = prompt_indices("  > ", len(remaining))
    return target_name, [remaining[i]["name"] for i in subtract_idxs]


def pick_prompt(clip_dir, target_name):
    prompts_path = os.path.join(clip_dir, "prompts.json")
    candidates = []
    if os.path.exists(prompts_path):
        with open(prompts_path) as f:
            candidates = json.load(f).get(target_name, [])

    if candidates:
        print(f"\n  Generated prompt candidates for '{target_name}':")
        for i, c in enumerate(candidates, 1):
            preview = c["prompt"][:100] + ("..." if len(c["prompt"]) > 100 else "")
            print(f"    {i}. [{c['variation']}] {preview}")
        write_own_idx = len(candidates) + 1
        print(f"    {write_own_idx}. Write my own")
        choice = prompt_index("  > ", write_own_idx)
        if choice < len(candidates):
            return candidates[choice]["prompt"]

    print("\n  Enter a Cosmos Transfer prompt describing the scene AFTER the change:")
    return input("  > ").strip()


def review_clip(clip_name, clip_dir, job, pipeline) -> bool:
    print("\n" + "=" * 60)
    print(f"Clip: {clip_name}")
    print("=" * 60)

    with open(os.path.join(clip_dir, "ontology.json")) as f:
        ontology_result = json.load(f)
    ontology = {"variation_ontology": ontology_result["variation_ontology"]}
    key_objects = ontology["variation_ontology"].get("key_objects", [])
    if not key_objects:
        print("  ⚠ No key objects in ontology — skipping.")
        return False

    vis_path = os.path.join(clip_dir, "_visual_qa.jpg")
    if os.path.exists(vis_path):
        print(f"  Visual QA image (inspect before picking): {vis_path}")

    target_name, subtract_names = pick_target_and_subtract(key_objects)
    prompt_text = pick_prompt(clip_dir, target_name)
    if not prompt_text:
        print("  ⚠ No prompt given — skipping this clip.")
        return False

    print("\n[Segmentation] Running SAM2...")
    seg_mask_path = run_segmentation_pipeline(
        video_path=ontology_result["clip_path"],
        ontology=ontology,
        target_name=target_name,
        subtract_names=subtract_names,
        bbox_frame_idx=ontology_result.get("frame_index", 0),
        sam2_checkpoint=job.sam2_checkpoint,
        sam2_config=job.sam2_config,
        output_dir=os.path.join(clip_dir, "seg_masks"),
        device=pipeline.device,
        num_points=job.sam2_num_points,
    )

    print("\n[Transfer] Running Cosmos Transfer inference...")
    spec = build_spec(
        name=clip_name,
        prompt=prompt_text,
        video_path=ontology_result["clip_path"],
        guidance=job.guidance,
        weights=job.weights,
        seg_mask_path=seg_mask_path,
    )
    spec_path = os.path.join(clip_dir, "spec.json")
    write_spec(spec, spec_path)

    transfer_output_dir = os.path.join(clip_dir, "transfer_output")
    success = run_inference(
        spec_path=spec_path,
        output_dir=transfer_output_dir,
        cosmos_repo_path=job.cosmos_transfer_repo,
        disable_guardrails=job.disable_guardrails,
    )

    if not success:
        return False

    with open(os.path.join(transfer_output_dir, ".done"), "w") as f:
        f.write(f"target={target_name}\nsubtract={subtract_names}\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Phase 2: interactive per-clip object picking, SAM2, Transfer")
    parser.add_argument("--job", "-j", required=True, help="Batch job config YAML path")
    parser.add_argument("--pipeline-config", default="config/default.yaml", help="Global pipeline config YAML path")
    args = parser.parse_args()

    pipeline = load_pipeline_config(args.pipeline_config)
    job = load_batch_job_config(args.job)
    resolve_job_paths(job, pipeline)

    pending = discover_pending_clips(job.output_dir)
    if not pending:
        print(f"Nothing to review under {job.output_dir}.")
        print("Run scripts/run_batch_prepare.py first, or every clip is already done.")
        return

    print(f"{len(pending)} clip(s) ready for review.")

    done, skipped = 0, 0
    for clip_name, clip_dir in pending:
        try:
            ok = review_clip(clip_name, clip_dir, job, pipeline)
        except KeyboardInterrupt:
            print("\n\nStopped. Re-run this command to resume where you left off.")
            break
        if ok:
            done += 1
        else:
            skipped += 1

    print(f"\n{'=' * 60}")
    print(f"Session complete: {done} clip(s) transferred, {skipped} skipped/failed")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
