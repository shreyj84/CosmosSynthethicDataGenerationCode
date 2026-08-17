"""Utility functions for video processing and file I/O."""

import cv2
import os
import json
import subprocess
from typing import Tuple, Dict, Any


def extract_frames(video_path: str, frames_dir: str) -> Tuple[int, float]:
    """
    Extract all frames from a video as numbered JPEGs.

    Args:
        video_path: Path to the input video file.
        frames_dir: Directory to save extracted frames.

    Returns:
        Tuple of (frame_count, fps).
    """
    os.makedirs(frames_dir, exist_ok=True)

    # Clear old frames
    for f in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, f))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(os.path.join(frames_dir, f"{frame_count:05d}.jpg"), frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames at {fps:.1f} FPS from {video_path}")
    return frame_count, fps


def get_video_resolution(video_path: str) -> Tuple[int, int]:
    """Return (width, height) of a video."""
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


def load_ontology(json_path: str) -> Dict[str, Any]:
    """Load a Cosmos Reason variation ontology JSON."""
    with open(json_path) as f:
        return json.load(f)


def get_objects_by_name(ontology: Dict[str, Any]) -> Dict[str, Dict]:
    """
    Extract a name → bbox_2d_pixels mapping from an ontology.

    Returns:
        Dict mapping object name to {'x1': int, 'y1': int, 'x2': int, 'y2': int}.
    """
    return {
        obj["name"]: obj["bbox_2d_pixels"]
        for obj in ontology["variation_ontology"]["key_objects"]
    }


def get_gpu_vram_mb() -> int:
    """Query GPU VRAM in MB via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )
        return int(result.stdout.strip())
    except Exception:
        return 0


def test_hf_download_speed(
    hf_token: str,
    test_url: str = "https://huggingface.co/nvidia/Cosmos-Predict2.5-2B/resolve/main/tokenizer.pth",
    timeout_s: int = 15,
) -> Dict[str, float]:
    """
    Test download speed to HuggingFace using both curl and wget.

    Returns:
        Dict with 'curl_bytes_per_sec' and 'wget_ok' keys.
    """
    # Test curl
    result = subprocess.run(
        [
            "curl", "-s", "-o", "/dev/null",
            "-w", "%{speed_download}",
            "--max-time", str(timeout_s),
            "-H", f"Authorization: Bearer {hf_token}",
            test_url,
        ],
        capture_output=True, text=True,
    )
    curl_speed = float(result.stdout.strip()) if result.stdout.strip() else 0

    # Test wget
    wget_result = subprocess.run(
        [
            "wget", "-q", f"--timeout={timeout_s}",
            "--header", f"Authorization: Bearer {hf_token}",
            "-O", "/dev/null",
            test_url,
        ],
        capture_output=True, text=True,
    )
    wget_ok = wget_result.returncode == 0

    print(f"curl speed:  {curl_speed:.0f} bytes/sec")
    print(f"wget works:  {'✅' if wget_ok else '❌'}")

    if curl_speed < 100_000 and wget_ok:
        print("→ Use wget-based downloads (curl is throttled)")
    elif curl_speed >= 1_000_000:
        print("→ Both methods work fine")

    return {"curl_bytes_per_sec": curl_speed, "wget_ok": wget_ok}
