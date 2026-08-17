"""
Phase 1 (unattended): raw multi-minute videos -> Cosmos Curate (split to Transfer's
recommended clip length/fps) -> Cosmos Reason (per-clip bbox + variation ontology) ->
GPT-4 Mini (candidate Cosmos Transfer prompts per detected object).

No human interaction happens in this script — it runs straight through every input
video and every clip Curate produces from it. Object/variation *selection* is a
separate, interactive step: scripts/run_interactive_review.py (phase 2).

Writes, per clip, a subdirectory under --job's output_dir:
    <output_dir>/<clip_name>/ontology.json     — Cosmos Reason output
    <output_dir>/<clip_name>/prompts.json      — { object_name: [{variation, prompt}] }
    <output_dir>/<clip_name>/_visual_qa.jpg    — bbox overlay for a quick sanity check
    <output_dir>/<clip_name>/_thinking.txt     — Cosmos Reason's raw reasoning trace

Usage:
    python scripts/run_batch_prepare.py --job config/jobs/bucket_cleaning_batch.yaml

    # Clips are already trimmed to length — skip the Curate split step:
    python scripts/run_batch_prepare.py --job config/jobs/bucket_cleaning_batch.yaml --skip-curate

    # Ontologies already exist from a previous run — only (re)generate prompts:
    python scripts/run_batch_prepare.py --job config/jobs/bucket_cleaning_batch.yaml --skip-curate --skip-reason
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_batch_job_config, load_pipeline_config, resolve_job_paths
from src.curate import split_video
from src.prompt_generator import generate_prompts_batch

VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".webm")


def collect_input_videos(input_path: str):
    if os.path.isdir(input_path):
        return sorted(
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.lower().endswith(VIDEO_EXTENSIONS)
        )
    return [input_path]


def generate_prompts_for_clip(clip_output_dir, ontology_result, job, openai_api_key):
    ontology = {"variation_ontology": ontology_result["variation_ontology"]}
    key_objects = ontology["variation_ontology"].get("key_objects", [])
    if not key_objects:
        print("  ⚠ no key objects in ontology, skipping prompt generation")
        return

    prompts_by_object = {}
    for obj in key_objects:
        name = obj.get("name")
        if not name:
            continue
        variations = obj.get("variations") or {}
        candidates = (
            list(variations.get("color_and_appearance", []))[: job.max_variations_per_object]
            + list(variations.get("material_and_texture", []))[: job.max_variations_per_object]
        )
        if not candidates:
            continue
        if not openai_api_key:
            print(f"  ⚠ {name}: no OpenAI key configured, skipping")
            continue

        print(f"  {name}: generating {len(candidates)} prompt candidate(s)")
        prompts_by_object[name] = generate_prompts_batch(
            ontology, name, candidates, api_key=openai_api_key, model=job.openai_model,
            temperature=job.openai_temperature, max_tokens=job.openai_max_tokens,
        )

    prompts_path = os.path.join(clip_output_dir, "prompts.json")
    with open(prompts_path, "w") as f:
        json.dump(prompts_by_object, f, indent=2)
    print(f"  Saved: {prompts_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1: curate, reason, and prompt-generate every clip, unattended")
    parser.add_argument("--job", "-j", required=True, help="Batch job config YAML path")
    parser.add_argument("--pipeline-config", default="config/default.yaml", help="Global pipeline config YAML path")
    parser.add_argument("--skip-curate", action="store_true",
                        help="Treat --job's input_video_path as already-trimmed clips (a directory or single file)")
    parser.add_argument("--skip-reason", action="store_true",
                        help="Reuse existing ontology.json per clip instead of re-running Cosmos Reason")
    args = parser.parse_args()

    pipeline = load_pipeline_config(args.pipeline_config)
    job = load_batch_job_config(args.job)
    resolve_job_paths(job, pipeline)

    hf_token = pipeline.hf_token or os.environ.get("HF_TOKEN", "")
    openai_api_key = job.openai_api_key or os.environ.get("OPENAI_API_KEY", "")

    print("=" * 60)
    print(f"Phase 1 (batch prepare): {job.job_name or os.path.basename(args.job)}")
    print(f"  Input:  {job.input_video_path}")
    print(f"  Output: {job.output_dir}")
    print("=" * 60)

    # Step 1: Curate — split every long video into clips
    if args.skip_curate:
        clip_paths = collect_input_videos(job.input_video_path)
        print(f"\n[Step 1] Skipping Curate — using {len(clip_paths)} pre-trimmed clip(s)")
    else:
        if not hf_token:
            print("❌ HF_TOKEN not set (env var or pipeline config) — needed for checkpoint access.")
            sys.exit(1)
        print("\n" + "=" * 60)
        print("[Step 1] Cosmos Curate — splitting")
        print("=" * 60)
        clip_paths = []
        for video_path in collect_input_videos(job.input_video_path):
            clip_paths.extend(split_video(
                input_video_path=video_path,
                output_clip_path=os.path.join(job.output_dir, "_clips"),
                cosmos_curate_repo=job.cosmos_curate_repo,
                model_weights_path=pipeline.checkpoint_dir,
                splitting_algorithm=job.curate_splitting_algorithm,
                threshold=job.curate_threshold,
                min_length_s=job.curate_min_length_s,
                max_length_s=job.curate_max_length_s,
                max_length_mode=job.curate_max_length_mode,
                num_download_workers=job.curate_num_download_workers,
                transcode_cpus=job.curate_transcode_cpus,
                clip_writer_workers=job.curate_clip_writer_workers,
            ))
        print(f"\n✅ Curate produced {len(clip_paths)} clip(s) total")

    if not clip_paths:
        print("❌ No clips to process.")
        sys.exit(1)

    # Step 2: Reason — load the model once, run it on every clip
    print("\n" + "=" * 60)
    print("[Step 2] Cosmos Reason")
    print("=" * 60)

    model, processor = None, None
    if not args.skip_reason:
        if not hf_token:
            print("❌ HF_TOKEN not set (env var or pipeline config).")
            sys.exit(1)
        from src.reason import load_reasoner
        processor, model = load_reasoner(job.reason_model_id, hf_token)

    ontology_results = {}
    failed = []
    for clip_path in clip_paths:
        clip_name = os.path.splitext(os.path.basename(clip_path))[0]
        clip_output_dir = os.path.join(job.output_dir, clip_name)

        if args.skip_reason:
            ontology_path = os.path.join(clip_output_dir, "ontology.json")
            if not os.path.exists(ontology_path):
                print(f"  ⚠ {clip_name}: no existing ontology.json, skipping (run without --skip-reason first)")
                continue
            with open(ontology_path) as f:
                ontology_results[clip_name] = (clip_output_dir, json.load(f))
            continue

        from src.reason import run_reason_on_clip
        try:
            result = run_reason_on_clip(
                clip_path, clip_output_dir, model, processor,
                num_objects=job.reason_num_objects,
                max_new_tokens=job.reason_max_new_tokens,
                frame_n_samples=job.reason_frame_samples,
                frame_window_size=job.reason_frame_window,
            )
            ontology_results[clip_name] = (clip_output_dir, result)
        except Exception as e:
            print(f"  ❌ {clip_name}: Reason failed — {e}")
            failed.append(clip_name)

    if model is not None:
        del model, processor
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # Step 3: Prompts — GPT-4 Mini turns each object's suggested variations into
    # candidate Cosmos Transfer prompts
    print("\n" + "=" * 60)
    print("[Step 3] Prompt generation")
    print("=" * 60)

    if not openai_api_key:
        print("  ⚠ No OpenAI API key configured — skipping prompt generation for all clips.")
        print("    Set OPENAI_API_KEY, or add prompts manually before running phase 2.")
    else:
        for clip_name, (clip_output_dir, result) in ontology_results.items():
            print(f"\n{clip_name}:")
            try:
                generate_prompts_for_clip(clip_output_dir, result, job, openai_api_key)
            except Exception as e:
                print(f"  ❌ {clip_name}: prompt generation failed — {e}")
                failed.append(clip_name)

    print("\n" + "=" * 60)
    print(f"✅ Phase 1 complete — {len(ontology_results)}/{len(clip_paths)} clip(s) ready for review")
    if failed:
        print(f"⚠ {len(set(failed))} clip(s) had errors: {sorted(set(failed))}")
    print(f"   Next: python scripts/run_interactive_review.py --job {args.job}")
    print("=" * 60)


if __name__ == "__main__":
    main()
