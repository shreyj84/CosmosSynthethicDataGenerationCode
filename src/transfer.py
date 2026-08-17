"""
Cosmos Transfer 2.5 wrapper — spec building and inference execution.

Handles:
    - Building valid spec JSON files from config
    - Running inference via subprocess
    - Weight normalization warnings
"""

import json
import os
import subprocess
from typing import Optional, Dict, Any

from .config import TransferWeights


def build_spec(
    name: str,
    prompt: str,
    video_path: str,
    guidance: int = 4,
    weights: Optional[TransferWeights] = None,
    seg_mask_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a Cosmos Transfer spec dictionary.

    Args:
        name: Job name (used as output filename).
        prompt: Text prompt describing the desired output.
        video_path: Path to the input video.
        guidance: Prompt guidance strength (2-7, default 4).
        weights: Control weights. Defaults to balanced preset.
        seg_mask_path: Path to segmentation mask video. None to skip seg.

    Returns:
        Spec dictionary ready to serialize as JSON.
    """
    if weights is None:
        weights = TransferWeights()

    # Validate and warn
    weights.validate()

    spec = {
        "name": name,
        "prompt": prompt,
        "video_path": video_path,
        "guidance": guidance,
        "edge": {"control_weight": weights.edge},
        "vis": {"control_weight": weights.vis},
        "depth": {"control_weight": weights.depth},
    }

    if seg_mask_path:
        if not os.path.exists(seg_mask_path):
            print(f"  ⚠ WARNING: Seg mask not found: {seg_mask_path}")
        spec["seg"] = {
            "control_weight": weights.seg,
            "control_path": seg_mask_path,
            "mask_path": seg_mask_path,
        }

    return spec


def write_spec(spec: Dict[str, Any], output_path: str) -> str:
    """Write a spec dict to a JSON file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(spec, f, indent=2)
    print(f"  Spec written: {output_path}")
    return output_path


def run_inference(
    spec_path: str,
    output_dir: str,
    cosmos_repo_path: str,
    disable_guardrails: bool = True,
    venv_python: Optional[str] = None,
) -> bool:
    """
    Run Cosmos Transfer inference.

    Args:
        spec_path: Path to the spec JSON file.
        output_dir: Directory for generated video output.
        cosmos_repo_path: Path to the cosmos-transfer2.5 repo.
        disable_guardrails: Whether to add --disable-guardrails flag.
        venv_python: Path to the venv Python binary. If None, uses
                     {cosmos_repo_path}/.venv/bin/python.

    Returns:
        True if inference succeeded, False otherwise.
    """
    if venv_python is None:
        venv_python = os.path.join(cosmos_repo_path, ".venv", "bin", "python")

    inference_script = os.path.join(cosmos_repo_path, "examples", "inference.py")

    if not os.path.exists(venv_python):
        raise FileNotFoundError(f"Cosmos venv Python not found: {venv_python}")
    if not os.path.exists(inference_script):
        raise FileNotFoundError(f"Inference script not found: {inference_script}")

    cmd = [
        venv_python,
        inference_script,
        "-i", spec_path,
        "-o", output_dir,
    ]
    if disable_guardrails:
        cmd.append("--disable-guardrails")

    print(f"\n{'=' * 60}")
    print(f"Running Cosmos Transfer inference")
    print(f"  Spec:   {spec_path}")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(cmd, cwd=cosmos_repo_path)

    if result.returncode == 0:
        print(f"\n✅ Inference complete: {output_dir}")
        return True
    else:
        print(f"\n❌ Inference failed with return code {result.returncode}")
        return False


def run_batch_inference(
    specs: list,
    cosmos_repo_path: str,
    base_output_dir: str,
    disable_guardrails: bool = True,
    venv_python: Optional[str] = None,
) -> Dict[str, bool]:
    """
    Run inference on multiple specs sequentially.

    Args:
        specs: List of (spec_dict, output_subdir) tuples.
        cosmos_repo_path: Path to cosmos-transfer2.5 repo.
        base_output_dir: Base directory for outputs.
        disable_guardrails: Whether to disable guardrails.
        venv_python: Path to venv Python binary.

    Returns:
        Dict mapping spec name → success bool.
    """
    results = {}

    for i, (spec, subdir) in enumerate(specs):
        name = spec["name"]
        print(f"\n{'#' * 60}")
        print(f"  Job {i + 1}/{len(specs)}: {name}")
        print(f"{'#' * 60}")

        output_dir = os.path.join(base_output_dir, subdir)
        spec_path = os.path.join(output_dir, "spec.json")
        write_spec(spec, spec_path)

        success = run_inference(
            spec_path, output_dir, cosmos_repo_path,
            disable_guardrails, venv_python,
        )
        results[name] = success

    # Summary
    print(f"\n{'=' * 60}")
    print("Batch Results:")
    for name, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {name}")
    print(f"{'=' * 60}")

    return results
