"""
Segmentation mask generation using SAM2 with bounding box subtraction.

This module takes bounding boxes (from Cosmos Reason variation ontology)
and produces binary mask videos suitable for Cosmos Transfer's seg control.

Pipeline:
    1. Load bounding boxes from ontology JSON
    2. Generate pixel-level masks on the reference frame using SAM2 image predictor
    3. Track masks through all frames using SAM2 video predictor (forward + backward)
    4. Subtract protected objects per-frame
    5. Export as seg_mask.mp4 (binary) and seg_overlay.mp4 (visualization)
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor


def load_bboxes_from_ontology(
    ontology: dict,
    target_name: str,
    subtract_names: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract bounding boxes from a Cosmos Reason ontology.

    Args:
        ontology: Loaded ontology dict (from JSON).
        target_name: Name of the object to change (e.g. "White surface").
        subtract_names: Names of objects to protect (e.g. ["White plastic bucket"]).

    Returns:
        Tuple of (target_boxes, subtract_boxes) as float tensors of shape (N, 4).
    """
    objects_by_name = {
        obj["name"]: obj["bbox_2d_pixels"]
        for obj in ontology["variation_ontology"]["key_objects"]
    }

    # Validate names exist
    if target_name not in objects_by_name:
        available = list(objects_by_name.keys())
        raise KeyError(
            f"Target object '{target_name}' not found. Available: {available}"
        )

    for name in subtract_names:
        if name not in objects_by_name:
            available = list(objects_by_name.keys())
            raise KeyError(
                f"Subtract object '{name}' not found. Available: {available}"
            )

    # Build target tensor
    t = objects_by_name[target_name]
    target_boxes = torch.tensor(
        [[t["x1"], t["y1"], t["x2"], t["y2"]]], dtype=torch.float32
    )

    # Build subtract tensor
    if subtract_names:
        subtract_list = []
        for name in subtract_names:
            b = objects_by_name[name]
            subtract_list.append([b["x1"], b["y1"], b["x2"], b["y2"]])
        subtract_boxes = torch.tensor(subtract_list, dtype=torch.float32)
    else:
        subtract_boxes = torch.tensor([], dtype=torch.float32).reshape(0, 4)

    return target_boxes, subtract_boxes


def generate_masks_on_frame(
    sam2_checkpoint: str,
    sam2_config: str,
    frame: np.ndarray,
    target_boxes: torch.Tensor,
    subtract_boxes: torch.Tensor,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate SAM2 masks with subtraction on a single frame.

    Args:
        sam2_checkpoint: Path to SAM2 checkpoint file.
        sam2_config: SAM2 config name (e.g. "sam2_hiera_l.yaml").
        frame: The reference frame as a numpy array (H, W, 3).
        target_boxes: Bounding boxes for the target object(s), shape (N, 4).
        subtract_boxes: Bounding boxes for objects to subtract, shape (M, 4).
        device: "cuda" or "cpu".

    Returns:
        Tuple of (target_masks, subtract_masks, final_mask).
        target_masks: shape (N, H, W), per-object target masks.
        subtract_masks: shape (M, H, W), per-object subtract masks. Empty if M=0.
        final_mask: shape (H, W), boolean mask after subtraction.
    """
    sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=device)
    image_predictor = SAM2ImagePredictor(sam2_model)
    image_predictor.set_image(frame)

    # Target masks
    target_masks, _, _ = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=target_boxes.to(device),
        multimask_output=False,
    )
    if target_masks.ndim == 4:
        target_masks = target_masks.squeeze(1)

    target_combined = np.any(target_masks, axis=0)

    # Subtract masks
    if len(subtract_boxes) > 0:
        subtract_masks, _, _ = image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=subtract_boxes.to(device),
            multimask_output=False,
        )
        if subtract_masks.ndim == 4:
            subtract_masks = subtract_masks.squeeze(1)
        subtract_combined = np.any(subtract_masks, axis=0)
    else:
        h, w = frame.shape[:2]
        subtract_masks = np.zeros((0, h, w), dtype=bool)
        subtract_combined = np.zeros((h, w), dtype=bool)

    # Subtract
    final_mask = target_combined & ~subtract_combined

    print(f"  Target pixels:   {target_combined.sum():,}")
    print(f"  Subtract pixels: {subtract_combined.sum():,}")
    print(f"  Final pixels:    {final_mask.sum():,}")

    return target_masks, subtract_masks, final_mask


def track_masks_through_video(
    sam2_checkpoint: str,
    sam2_config: str,
    frames_dir: str,
    masks: np.ndarray,
    bbox_frame_idx: int = 0,
    num_points: int = 10,
    device: str = "cuda",
) -> Dict[int, np.ndarray]:
    """
    Track a set of masks through all video frames using SAM2 video predictor.

    Propagates both forward and backward from the reference frame.

    Args:
        sam2_checkpoint: Path to SAM2 checkpoint.
        sam2_config: SAM2 config name.
        frames_dir: Directory containing numbered JPEG frames.
        masks: Masks from the reference frame, shape (N, H, W).
        bbox_frame_idx: Which frame the masks were generated on.
        num_points: Number of sample points per mask for tracking.
        device: "cuda" or "cpu".

    Returns:
        Dict mapping frame_idx → masks array of shape (N, H, W).
    """
    video_predictor = build_sam2_video_predictor(
        sam2_config, sam2_checkpoint, device=device
    )
    inference_state = video_predictor.init_state(video_path=frames_dir)

    registered = 0
    for obj_idx in range(len(masks)):
        mask = masks[obj_idx]
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        y_coords, x_coords = np.where(mask > 0)
        if len(y_coords) == 0:
            continue

        n_pts = min(num_points, len(y_coords))
        indices = np.linspace(0, len(y_coords) - 1, n_pts, dtype=int)
        points = np.stack([x_coords[indices], y_coords[indices]], axis=1)
        point_labels = np.ones(n_pts, dtype=np.int32)

        video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=bbox_frame_idx,
            obj_id=obj_idx,
            points=points,
            labels=point_labels,
        )
        registered += 1

    print(f"  Registered {registered} objects on frame {bbox_frame_idx}")

    segments = {}

    # Forward propagation
    for frame_idx, obj_ids, mask_logits in video_predictor.propagate_in_video(
        inference_state
    ):
        frame_masks = (mask_logits > 0.0).cpu().numpy().squeeze(1)
        segments[frame_idx] = frame_masks

    # Backward propagation
    for frame_idx, obj_ids, mask_logits in video_predictor.propagate_in_video(
        inference_state, reverse=True
    ):
        frame_masks = (mask_logits > 0.0).cpu().numpy().squeeze(1)
        segments[frame_idx] = frame_masks

    print(f"  Tracked across {len(segments)} frames (forward + backward)")
    return segments


def export_mask_videos(
    target_segments: Dict[int, np.ndarray],
    subtract_segments: Dict[int, np.ndarray],
    frames_dir: str,
    output_dir: str,
    fps: float,
    frame_count: int,
) -> Tuple[str, str]:
    """
    Export the per-frame subtracted masks as video files.

    Args:
        target_segments: Frame index → target masks array.
        subtract_segments: Frame index → subtract masks array.
        frames_dir: Directory with source JPEG frames.
        output_dir: Where to save output videos.
        fps: Frame rate for output videos.
        frame_count: Total number of frames.

    Returns:
        Tuple of (mask_path, overlay_path).
    """
    os.makedirs(output_dir, exist_ok=True)

    sample_frame = cv2.imread(os.path.join(frames_dir, "00000.jpg"))
    h, w = sample_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    mask_path = os.path.join(output_dir, "seg_mask.mp4")
    overlay_path = os.path.join(output_dir, "seg_overlay.mp4")

    mask_writer = cv2.VideoWriter(mask_path, fourcc, fps, (w, h), isColor=False)
    overlay_writer = cv2.VideoWriter(overlay_path, fourcc, fps, (w, h))

    for frame_idx in range(frame_count):
        # Target mask
        tgt_mask = np.zeros((h, w), dtype=bool)
        if frame_idx in target_segments:
            tgt_mask = np.any(target_segments[frame_idx], axis=0)

        # Subtract mask
        sub_mask = np.zeros((h, w), dtype=bool)
        if frame_idx in subtract_segments:
            sub_mask = np.any(subtract_segments[frame_idx], axis=0)

        # Subtract
        final = (tgt_mask & ~sub_mask).astype(np.uint8) * 255
        mask_writer.write(final)

        # Overlay
        orig = cv2.imread(os.path.join(frames_dir, f"{frame_idx:05d}.jpg"))
        overlay = orig.copy()
        overlay[final > 0] = (
            overlay[final > 0] * 0.5
            + np.array([0, 255, 0], dtype=np.float64) * 0.5
        ).astype(np.uint8)
        overlay_writer.write(overlay)

    mask_writer.release()
    overlay_writer.release()

    print(f"  Saved: {mask_path}")
    print(f"  Saved: {overlay_path}")
    return mask_path, overlay_path


def run_segmentation_pipeline(
    video_path: str,
    ontology: dict,
    target_name: str,
    subtract_names: List[str],
    bbox_frame_idx: int,
    sam2_checkpoint: str,
    sam2_config: str,
    output_dir: str,
    frames_dir: Optional[str] = None,
    device: str = "cuda",
    num_points: int = 10,
) -> str:
    """
    Run the full segmentation pipeline end-to-end.

    Args:
        video_path: Path to input video.
        ontology: Loaded Cosmos Reason ontology dict.
        target_name: Object to change.
        subtract_names: Objects to protect.
        bbox_frame_idx: Frame index where bounding boxes were generated.
        sam2_checkpoint: Path to SAM2 checkpoint.
        sam2_config: SAM2 config name.
        output_dir: Where to save mask videos.
        frames_dir: Scratch directory for extracted frames. Defaults to a
            subdirectory of output_dir (not a shared /tmp path) so concurrent runs
            against different clips never collide.
        device: "cuda" or "cpu".
        num_points: Sample points per mask when seeding SAM2's video tracking.

    Returns:
        Path to the generated seg_mask.mp4.
    """
    from .utils import extract_frames

    if frames_dir is None:
        frames_dir = os.path.join(output_dir, "_scratch", "frames")

    print("=" * 60)
    print(f"Segmentation Pipeline: {os.path.basename(video_path)}")
    print(f"  Target: {target_name}")
    print(f"  Subtract: {subtract_names}")
    print(f"  Reference frame: {bbox_frame_idx}")
    print("=" * 60)

    # Step 1: Extract frames
    print("\n[1/5] Extracting frames...")
    frame_count, fps = extract_frames(video_path, frames_dir)

    # Step 2: Load bounding boxes
    print("\n[2/5] Loading bounding boxes...")
    target_boxes, subtract_boxes = load_bboxes_from_ontology(
        ontology, target_name, subtract_names
    )

    # Step 3: Generate masks on reference frame
    print("\n[3/5] Generating masks on reference frame...")
    frame_path = os.path.join(frames_dir, f"{bbox_frame_idx:05d}.jpg")
    frame = np.array(Image.open(frame_path))

    target_masks, subtract_masks, final_mask = generate_masks_on_frame(
        sam2_checkpoint, sam2_config, frame, target_boxes, subtract_boxes, device
    )

    # Step 4: Track through video
    print("\n[4/5] Tracking through video...")
    print("  Tracking target...")
    target_segments = track_masks_through_video(
        sam2_checkpoint, sam2_config, frames_dir,
        target_masks, bbox_frame_idx, num_points=num_points, device=device,
    )

    if len(subtract_masks) > 0:
        print("  Tracking subtract objects...")
        subtract_segments = track_masks_through_video(
            sam2_checkpoint, sam2_config, frames_dir,
            subtract_masks, bbox_frame_idx, num_points=num_points, device=device,
        )
    else:
        subtract_segments = {}

    # Step 5: Export
    print("\n[5/5] Exporting mask videos...")
    mask_path, overlay_path = export_mask_videos(
        target_segments, subtract_segments,
        frames_dir, output_dir, fps, frame_count,
    )

    print(f"\n✅ Segmentation complete: {mask_path}")
    return mask_path
