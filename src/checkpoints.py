"""
Checkpoint download and caching.

Cosmos-Predict2.5-2B and Cosmos-Transfer2.5-2B download via wget: the Python
HuggingFace client gets throttled on Google Colab but wget runs at full speed there.
This module downloads those two via wget and creates the HuggingFace cache structure
so inference finds them. (docs/setup_aws.md notes this throttling doesn't apply on
AWS — the wget path still works fine there, just isn't required.)

TransNetV2 (used by Cosmos Curate's splitting step) downloads via the native
huggingface_hub client instead, since there's no throttling concern to route around
for it. InternVideo2 (embeddings) and Qwen2.5-VL (captioning) are intentionally not
downloaded here — this pipeline runs Curate with --no-generate-embeddings
--no-generate-captions and doesn't use either model.
"""

import os
import json
import subprocess
import shutil
import tempfile
from typing import List, Optional

from .config import DATA_ROOT, SAM2_CHECKPOINT_URL

DEFAULT_WGET_TIMEOUT_S = 120
DEFAULT_HF_API_TIMEOUT_S = 30


def wget_hf_file(
    repo_id: str,
    filename: str,
    hf_token: str,
    local_dir: str,
    revision: str = "main",
    timeout_s: int = DEFAULT_WGET_TIMEOUT_S,
) -> Optional[str]:
    """Download a single file from HuggingFace using wget."""
    output_path = os.path.join(local_dir, filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  ✅ {filename} ({size_mb:.1f} MB) — cached")
        return output_path

    url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"
    print(f"  ⬇ {filename}...")

    result = subprocess.run([
        "wget", "-q", "--show-progress", f"--timeout={timeout_s}",
        "--header", f"Authorization: Bearer {hf_token}",
        "-O", output_path, url,
    ])

    if not os.path.exists(output_path):
        print(f"  ❌ {filename} — download failed")
        return None

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    if size_mb < 0.01:
        print(f"  ❌ {filename} — too small, likely error page")
        os.remove(output_path)
        return None

    print(f"  ✅ {filename} ({size_mb:.1f} MB)")
    return output_path


def download_hf_repo(
    repo_id: str,
    hf_token: str,
    local_dir: str,
    revision: str = "main",
    include_extensions: tuple = (".pt", ".pth"),
    exclude_prefixes: Optional[List[str]] = None,
    timeout_s: int = DEFAULT_WGET_TIMEOUT_S,
    api_timeout_s: int = DEFAULT_HF_API_TIMEOUT_S,
) -> str:
    """
    Download model files from a HuggingFace repo using wget.

    Args:
        repo_id: HuggingFace repo (e.g. "nvidia/Cosmos-Predict2.5-2B").
        hf_token: HuggingFace auth token.
        local_dir: Base directory for downloads.
        revision: Git revision to download.
        include_extensions: Only download files with these extensions.
        exclude_prefixes: Skip files starting with these prefixes. NOTE: this encodes
            the repo's current folder layout — if NVIDIA reorganizes it, these prefixes
            may stop matching anything (see the zero-files warning below) or start
            excluding files they shouldn't.
        timeout_s: wget timeout for the actual file downloads.
        api_timeout_s: wget timeout for the file-listing API call.

    Returns:
        Path to the local download directory.
    """
    print(f"\n{'=' * 60}")
    print(f"Downloading {repo_id}")
    print(f"{'=' * 60}")

    # Get file list from API
    api_url = f"https://huggingface.co/api/models/{repo_id}?revision={revision}"
    info_path = os.path.join(tempfile.gettempdir(), f"hf_repo_info_{repo_id.replace('/', '--')}.json")

    subprocess.run([
        "wget", "-q", f"--timeout={api_timeout_s}",
        "--header", f"Authorization: Bearer {hf_token}",
        "-O", info_path, api_url,
    ])

    with open(info_path) as f:
        data = json.load(f)

    files = [s["rfilename"] for s in data.get("siblings", [])]
    total_files = len(files)

    if include_extensions:
        files = [f for f in files if any(f.endswith(ext) for ext in include_extensions)]

    if exclude_prefixes:
        files = [
            f for f in files if not any(f.startswith(p) for p in exclude_prefixes)
        ]

    if total_files > 0 and not files:
        print(
            f"  ⚠ WARNING: {repo_id} has {total_files} file(s), but include_extensions="
            f"{include_extensions} / exclude_prefixes={exclude_prefixes} matched none of them. "
            f"The repo's layout may have changed — check https://huggingface.co/{repo_id} "
            f"and adjust these filters rather than assuming there's nothing to download."
        )

    print(f"Found {len(files)} files:")
    for f in files:
        print(f"  • {f}")

    repo_dir = os.path.join(local_dir, repo_id.replace("/", "--"))

    for filename in files:
        wget_hf_file(repo_id, filename, hf_token, repo_dir, revision, timeout_s=timeout_s)

    return repo_dir


def link_to_hf_cache(repo_id: str, download_dir: str, hf_token: str, api_timeout_s: int = DEFAULT_HF_API_TIMEOUT_S):
    """
    Create HuggingFace cache symlinks so inference finds wget'd files.

    Args:
        repo_id: HuggingFace repo ID.
        download_dir: Where the files were downloaded to.
        hf_token: HuggingFace auth token (to query current revision).
        api_timeout_s: wget timeout for the revision-lookup API call.
    """
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
    cache_dir = os.path.join(hf_cache, f"models--{repo_id.replace('/', '--')}")

    # Get current commit hash
    info_path = os.path.join(tempfile.gettempdir(), f"hf_repo_revision_{repo_id.replace('/', '--')}.json")
    subprocess.run([
        "wget", "-q", f"--timeout={api_timeout_s}",
        "--header", f"Authorization: Bearer {hf_token}",
        "-O", info_path,
        f"https://huggingface.co/api/models/{repo_id}",
    ])
    with open(info_path) as f:
        revision = json.load(f).get("sha", "main")

    # Create refs/main
    refs_dir = os.path.join(cache_dir, "refs")
    os.makedirs(refs_dir, exist_ok=True)
    with open(os.path.join(refs_dir, "main"), "w") as f:
        f.write(revision)

    # Symlink snapshot → download dir
    snapshot_dir = os.path.join(cache_dir, "snapshots", revision)
    if os.path.exists(snapshot_dir):
        if os.path.islink(snapshot_dir):
            os.unlink(snapshot_dir)
        else:
            shutil.rmtree(snapshot_dir)

    os.makedirs(os.path.dirname(snapshot_dir), exist_ok=True)
    os.symlink(download_dir, snapshot_dir)

    print(f"  ✅ {repo_id} → {revision[:12]}... → {download_dir}")


def link_specific_revision(repo_id: str, revision: str, download_dir: str):
    """
    Create a cache symlink for a specific revision hash.

    Use this when inference requests a pinned revision that differs
    from the current 'main'.
    """
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
    cache_dir = os.path.join(hf_cache, f"models--{repo_id.replace('/', '--')}")
    snapshot_dir = os.path.join(cache_dir, "snapshots", revision)

    if os.path.exists(snapshot_dir):
        if os.path.islink(snapshot_dir):
            os.unlink(snapshot_dir)

    os.makedirs(os.path.dirname(snapshot_dir), exist_ok=True)
    os.symlink(download_dir, snapshot_dir)

    print(f"  ✅ {revision[:16]}... → {download_dir}")


def download_transnetv2(hf_token: str, checkpoint_dir: str) -> str:
    """
    Download TransNetV2 (used by Cosmos Curate's splitting step) via the native
    huggingface_hub client — no wget hack needed, this isn't affected by the Colab
    throttling issue the rest of this module works around.
    """
    from huggingface_hub import snapshot_download

    local_dir = os.path.join(checkpoint_dir, "Sn4kehead--TransNetV2")
    if os.path.isdir(local_dir) and os.listdir(local_dir):
        print(f"  ✅ TransNetV2 — cached at {local_dir}")
        return local_dir

    print(f"\n{'=' * 60}")
    print("Downloading TransNetV2 (Cosmos Curate splitting)")
    print(f"{'=' * 60}")
    snapshot_download(repo_id="Sn4kehead/TransNetV2", local_dir=local_dir, token=hf_token)
    print(f"  ✅ TransNetV2 → {local_dir}")
    return local_dir


def download_all_checkpoints(
    hf_token: str,
    checkpoint_dir: Optional[str] = None,
    sam2_checkpoint_url: str = SAM2_CHECKPOINT_URL,
    predict_exclude_prefixes: Optional[List[str]] = None,
    transfer_exclude_prefixes: Optional[List[str]] = None,
    timeout_s: int = DEFAULT_WGET_TIMEOUT_S,
    api_timeout_s: int = DEFAULT_HF_API_TIMEOUT_S,
):
    """
    Download all checkpoints needed for the full pipeline.

    Does NOT download nvidia/Cosmos3-Nano (Cosmos Reason) — that loads lazily via
    transformers' own from_pretrained() caching the first time src/reason.py's
    load_reasoner() is called, since it's a standard (non-throttled-on-AWS) HF
    download with no need for the wget workaround.

    Args:
        hf_token: HuggingFace auth token.
        checkpoint_dir: Base directory for all downloads. Defaults to
            {DATA_ROOT}/checkpoints (see src/config.py) if not given.
        sam2_checkpoint_url: Direct download URL for the SAM2 checkpoint. Defaults to
            src/config.py's SAM2_CHECKPOINT_URL (the one other place this pipeline
            needs it — src/config.py:PipelineConfig — reads the same constant).
        predict_exclude_prefixes / transfer_exclude_prefixes: HF file-path prefixes to
            skip for each repo. Defaults match this pipeline's original filtering;
            override if NVIDIA's repo layout changes (see download_hf_repo's warning
            when a filter matches nothing).
        timeout_s / api_timeout_s: wget timeouts for file downloads / API calls.
    """
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(DATA_ROOT, "checkpoints")
    if predict_exclude_prefixes is None:
        predict_exclude_prefixes = ["auto/", "robot/", "base/pre-trained/", "base/post-trained/"]
    if transfer_exclude_prefixes is None:
        transfer_exclude_prefixes = ["auto/", "robot/"]

    # Cosmos Predict2.5-2B (base model + tokenizer)
    predict_dir = download_hf_repo(
        "nvidia/Cosmos-Predict2.5-2B",
        hf_token,
        checkpoint_dir,
        exclude_prefixes=predict_exclude_prefixes,
        timeout_s=timeout_s,
        api_timeout_s=api_timeout_s,
    )

    # Cosmos Transfer2.5-2B (control-net weights)
    transfer_dir = download_hf_repo(
        "nvidia/Cosmos-Transfer2.5-2B",
        hf_token,
        checkpoint_dir,
        exclude_prefixes=transfer_exclude_prefixes,
        timeout_s=timeout_s,
        api_timeout_s=api_timeout_s,
    )

    # Video Depth Anything (for auto depth maps)
    depth_path = os.path.join(checkpoint_dir, "video_depth_anything_vits.pth")
    if not os.path.exists(depth_path):
        print(f"\n{'=' * 60}")
        print("Downloading Video Depth Anything")
        print(f"{'=' * 60}")
        subprocess.run([
            "wget", "-q", "--show-progress", f"--timeout={timeout_s}",
            "-O", depth_path,
            "https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth",
        ])
    else:
        print("\n✅ Video Depth Anything already cached")

    # SAM2 checkpoint
    sam2_path = os.path.join(checkpoint_dir, "sam2_hiera_large.pt")
    if not os.path.exists(sam2_path):
        print(f"\n{'=' * 60}")
        print("Downloading SAM2 Hiera-Large")
        print(f"{'=' * 60}")
        subprocess.run([
            "wget", "-q", "--show-progress", f"--timeout={timeout_s}",
            "-O", sam2_path,
            sam2_checkpoint_url,
        ])
    else:
        print("\n✅ SAM2 already cached")

    # TransNetV2 (Cosmos Curate splitting)
    transnetv2_dir = download_transnetv2(hf_token, checkpoint_dir)

    # Link to HF cache
    print(f"\n{'=' * 60}")
    print("Linking to HuggingFace cache")
    print(f"{'=' * 60}")
    link_to_hf_cache("nvidia/Cosmos-Predict2.5-2B", predict_dir, hf_token, api_timeout_s=api_timeout_s)
    link_to_hf_cache("nvidia/Cosmos-Transfer2.5-2B", transfer_dir, hf_token, api_timeout_s=api_timeout_s)

    print(f"\n{'=' * 60}")
    print("All checkpoints ready.")
    print(f"{'=' * 60}")

    return {
        "predict_dir": predict_dir,
        "transfer_dir": transfer_dir,
        "sam2_checkpoint": sam2_path,
        "transnetv2_dir": transnetv2_dir,
    }


def save_to_persistent_storage(checkpoint_dir: str, storage_path: str):
    """Copy checkpoints to persistent storage (Drive, EBS, etc.)."""
    if not os.path.exists(storage_path):
        print(f"Saving checkpoints to {storage_path}...")
        shutil.copytree(checkpoint_dir, storage_path)
        print(f"✅ Saved ({_dir_size_gb(storage_path):.1f} GB)")
    else:
        print(f"✅ Persistent cache exists: {storage_path}")


def restore_from_persistent_storage(storage_path: str, checkpoint_dir: str):
    """Restore checkpoints from persistent storage."""
    if os.path.exists(storage_path):
        print(f"Restoring from {storage_path}...")
        shutil.copytree(storage_path, checkpoint_dir, dirs_exist_ok=True)
        print(f"✅ Restored ({_dir_size_gb(checkpoint_dir):.1f} GB)")
    else:
        print(f"❌ No cache at {storage_path}. Run download first.")


def _dir_size_gb(path: str) -> float:
    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / (1024 ** 3)
