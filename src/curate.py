"""
Cosmos Curate wrapper — splits long, multi-minute source videos into clips at the
length/frame rate Cosmos Transfer expects.

Runs `cosmos_curate/pipelines/video/run_pipeline.py split` via subprocess inside the
cosmos-curate repo's pixi environment, the same pattern src/transfer.py uses for the
cosmos-transfer2.5 repo's venv. Building the command as an argv list (rather than a
shell string) sidesteps the classic bash line-continuation footgun — a stray comment
after a trailing backslash silently breaks the command.
"""

import glob
import os
import subprocess
from typing import List, Optional


def run_split(
    input_video_path: str,
    output_clip_path: str,
    cosmos_curate_repo: str,
    model_weights_path: str,
    splitting_algorithm: str = "transnetv2",
    threshold: float = 0.2,
    min_length_s: float = 2.0,
    max_length_s: float = 4.0,
    max_length_mode: str = "stride",
    num_download_workers: int = 2,
    transcode_cpus: int = 2,
    clip_writer_workers: int = 2,
    generate_embeddings: bool = False,
    generate_captions: bool = False,
    verbose: bool = True,
) -> bool:
    """
    Run Cosmos Curate's video-splitting pipeline on one input video (or a directory
    of videos — the pipeline handles both).

    Embeddings (InternVideo2) and captions (Qwen2.5-VL) default to disabled — this
    pipeline doesn't use either, only the TransNetV2-based split itself.

    Returns:
        True if the pipeline exited successfully.
    """
    script = os.path.join(cosmos_curate_repo, "cosmos_curate", "pipelines", "video", "run_pipeline.py")
    if not os.path.exists(script):
        raise FileNotFoundError(f"Cosmos Curate pipeline script not found: {script}")

    os.makedirs(output_clip_path, exist_ok=True)

    cmd = [
        "pixi", "run", "python", script, "split",
        "--input-video-path", input_video_path,
        "--output-clip-path", output_clip_path,
        "--splitting-algorithm", splitting_algorithm,
    ]

    if splitting_algorithm == "transnetv2":
        cmd += [
            "--transnetv2-threshold", str(threshold),
            "--transnetv2-min-length-s", str(min_length_s),
            "--transnetv2-max-length-s", str(max_length_s),
            "--transnetv2-max-length-mode", max_length_mode,
        ]

    cmd += [
        "--model-weights-path", model_weights_path,
        "--transcode-cpus-per-worker", str(transcode_cpus),
        "--num-download-workers-per-node", str(num_download_workers),
        "--num-clip-writer-workers-per-node", str(clip_writer_workers),
    ]

    if not generate_embeddings:
        cmd.append("--no-generate-embeddings")
    if not generate_captions:
        cmd.append("--no-generate-captions")
    if verbose:
        cmd.append("--verbose")

    env = os.environ.copy()
    env["PYTHONPATH"] = cosmos_curate_repo

    print(f"\n{'=' * 60}")
    print("Running Cosmos Curate split")
    print(f"  Input:  {input_video_path}")
    print(f"  Output: {output_clip_path}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(cmd, cwd=cosmos_curate_repo, env=env)

    if result.returncode == 0:
        print(f"\n✅ Curate split complete: {output_clip_path}")
        return True
    else:
        print(f"\n❌ Curate split failed with return code {result.returncode}")
        return False


def find_output_clips(output_clip_path: str) -> List[str]:
    """Return every .mp4 clip Curate produced, sorted for deterministic ordering."""
    pattern = os.path.join(output_clip_path, "**", "*.mp4")
    return sorted(glob.glob(pattern, recursive=True))


def split_video(
    input_video_path: str,
    output_clip_path: str,
    cosmos_curate_repo: str,
    model_weights_path: str,
    **split_kwargs,
) -> List[str]:
    """
    High-level entry point: run the split, then return the list of produced clips.

    Raises:
        RuntimeError if the split subprocess fails.
    """
    ok = run_split(
        input_video_path=input_video_path,
        output_clip_path=output_clip_path,
        cosmos_curate_repo=cosmos_curate_repo,
        model_weights_path=model_weights_path,
        **split_kwargs,
    )
    if not ok:
        raise RuntimeError(f"Cosmos Curate split failed for {input_video_path}")

    clips = find_output_clips(output_clip_path)
    print(f"  Found {len(clips)} output clip(s)")
    return clips
