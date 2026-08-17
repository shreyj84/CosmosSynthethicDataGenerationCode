"""
Cosmos Reason wrapper — runs nvidia/Cosmos3-Nano on one representative frame per clip
to produce a variation ontology: key objects, bounding boxes, and suggested
color/material/texture variations for synthetic data generation.

Ported from a working Colab notebook cell (model loading, frame selection via optical
flow + sharpness, prompting, and JSON-repair parsing are all unchanged from that
snippet). Made reusable (load the model once, call run_reason_on_clip() per clip) and
headless-safe (matplotlib Agg backend — the QA overlay and per-object crops are saved
to disk, never displayed interactively, since batch runs happen on a server with no
display).

Output schema — result["variation_ontology"]["key_objects"][i] with `name`,
`bbox_2d_pixels: {x1,y1,x2,y2}`, `current_appearance`, `variations` — matches what
src/utils.py, src/segmentation.py, and src/prompt_generator.py already expect from a
hand-produced ontology JSON. This is a drop-in generator for that JSON.

NOTE: `nvidia/Cosmos3-Nano` and `Cosmos3OmniForConditionalGeneration` are very recent —
the exact `transformers`/`diffusers` versions needed to load them aren't pinned here.
Copy the working versions from the environment this was developed in (see
scripts/setup_environment.sh) rather than assuming latest-on-PyPI.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from json_repair import repair_json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Default number of objects requested per frame. In the original notebook this was a
# NUM_OBJECTS variable that was defined but never actually wired into the prompt text
# (which hardcoded "up to 10" regardless). build_bbox_variation_prompt() below now
# actually uses it — default matches the model's original real behavior (10), not the
# unused variable's old value (5), so nothing changes unless you explicitly override it.
DEFAULT_NUM_OBJECTS = 10

_COSMOS3_BBOX_VARIATION_PROMPT_TEMPLATE = """You are an expert robotics training data analyst with precise spatial grounding.

Analyze this egocentric image of a person performing a household task.

Your goals:
1. Identify the most task-relevant objects in the scene (up to __NUM_OBJECTS__).
   Only include objects that are directly involved in or essential to
   understanding the task.
2. Provide a precise 2D bounding box for each object
3. Enumerate EVERY possible variation for synthetic data generation

BOUNDING BOX FORMAT:
For each object, provide "bbox_2d": [x1, y1, x2, y2] using normalized
coordinates in the 0-1000 range (top-left corner is [0,0], bottom-right
is [1000,1000]). Be precise — the box should tightly enclose the object.

OUTPUT FORMAT — respond in this exact JSON:

{
  "location": {
    "room_type": "",
    "setting": "",
    "time_of_day": "",
    "lighting_condition": "",
    "room_style": ""
  },

  "task": {
    "name": "",
    "description": "",
    "phase": "",
    "action_primitive": "",
    "difficulty": "",
    "typical_duration_seconds": 0
  },

  "key_objects": [
    {
      "rank": 1,
      "name": "",
      "bbox_2d": [0, 0, 0, 0],
      "is_primary": true,
      "role_in_task": "",

      "current_appearance": {
        "color_description": "",
        "rgb": [0.0, 0.0, 0.0],
        "hex": "",
        "material": "",
        "texture": "",
        "condition": "",
        "size_estimate": ""
      },

      "current_state": {
        "position_on_surface": "",
        "height_above_surface": "",
        "orientation": "",
        "interaction_state": "",
        "task_completion_phase": ""
      },

      "variations": {
        "color_and_appearance": [],
        "material_and_texture": [],
        "size_and_shape": [],
        "state_and_position": [],
        "condition": [],
        "interaction": [],
        "brand_and_style": []
      },

      "robustness_gaps": {
        "what_robot_must_handle": [],
        "common_failure_modes": [],
        "edge_cases": []
      }
    }
  ],

  "scene_variations": {
    "lighting": [],
    "camera_viewpoint": [],
    "room_style_and_culture": [],
    "time_of_day": [],
    "background_clutter": [],
    "person_characteristics": [],
    "task_sequence_variations": []
  },

  "dataset_gaps": {
    "what_current_clip_shows": "",
    "what_is_missing": [],
    "highest_priority_variations": []
  }
}

INSTRUCTIONS:

BOUNDING BOXES: For EVERY object in key_objects, the "bbox_2d" field is
MANDATORY. Coordinates are [x1, y1, x2, y2] normalized to 0-1000 range.
x1,y1 is the top-left corner. x2,y2 is the bottom-right corner. Be tight
and precise. Objects partially out of frame should have boxes clipped to
the frame boundary (0 or 1000).

LOCATION: Be specific — not just "bedroom" but "small cluttered student
bedroom with dark carpet".

TASK: Identify the exact action primitive. "Making bed" is too broad —
"pulling bedspread from foot toward headboard with both hands" is correct.

KEY OBJECTS: For each, enumerate EVERY variation dimension exhaustively:

  color_and_appearance: every color family, pattern, print. Include 10+
    solid colors, patterns (striped, plaid, floral, geometric, plain),
    finishes (matte, glossy, metallic), worn/faded/stained versions.

  material_and_texture: every realistic material. Include tactile
    properties relevant to robot grasping.

  size_and_shape: every realistic size variation. Different aspect
    ratios. Folded vs unfolded. Stacked vs single.

  state_and_position: every position/orientation/arrangement during
    this task. Before/during/after each action step. Every failure state.

  condition: new, used, worn, dirty, wet, damaged, wrinkled, stained,
    torn, faded, dusty, sticky.

  interaction: how a robot gripper might interact. Grasping points,
    approach angles, grip types, force, fragility, slipperiness.

  brand_and_style: realistic brand/style variations that change
    appearance without changing function.

SCENE VARIATIONS: Be exhaustive for lighting, camera viewpoint,
room style, time of day, background clutter, person characteristics,
and task sequence variations.

DATASET GAPS: What does training ONLY on this clip fail to capture?
Rank by impact on robot policy robustness.

Answer the question in the following format:
<think>
your reasoning
</think>

<answer>
your JSON here
</answer>"""


def build_bbox_variation_prompt(num_objects: int = DEFAULT_NUM_OBJECTS) -> str:
    """Build the Cosmos Reason prompt, asking for up to `num_objects` objects."""
    return _COSMOS3_BBOX_VARIATION_PROMPT_TEMPLATE.replace("__NUM_OBJECTS__", str(num_objects))


# Default-object-count prompt, kept as a module-level constant for anyone importing it
# directly (e.g. quick scripts/notebooks) — equivalent to build_bbox_variation_prompt().
COSMOS3_BBOX_VARIATION_PROMPT = build_bbox_variation_prompt()


BOX_COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
]


def load_reasoner(model_id: str, hf_token: str, torch_dtype=torch.bfloat16, device_map: str = "auto"):
    """Load the Cosmos Reason model + processor once, reused across every clip in a run."""
    from transformers import AutoProcessor, Cosmos3OmniForConditionalGeneration

    print(f"Loading {model_id} reasoner... (first run downloads a large checkpoint)")
    processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
    model = Cosmos3OmniForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        token=hf_token,
    )
    print("✓ Reasoner loaded")
    if torch.cuda.is_available():
        print(f"  VRAM used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    return processor, model


def select_representative_frame(
    clip_path: str, n_samples: int = 30, window_size: int = 5
) -> Tuple[Optional[Image.Image], Optional[np.ndarray], int, int, int]:
    """
    Select the most representative frame using optical flow + sharpness.

    Strategy:
      1. Sample n_samples frames evenly across the clip
      2. Compute optical flow magnitude between each consecutive pair to find where
         peak motion (active manipulation) is happening
      3. Find the sliding window of `window_size` frames with the highest average
         motion — this is the "active task" region
      4. Within that window, pick the sharpest frame (Laplacian variance) so the
         model gets a crisp image for bounding box grounding

    Returns:
        (pil_image, frame_bgr, width, height, chosen_frame_idx)
    """
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        return None, None, 0, 0, -1

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    print(f"  Clip: {total_frames} frames @ {fps:.1f}fps ({duration:.1f}s)")

    if total_frames < 3:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None, None, 0, 0, -1
        h, w = frame.shape[:2]
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), frame, w, h, 0

    n_samples = min(n_samples, total_frames)
    sample_indices = [int(total_frames * i / n_samples) for i in range(n_samples)]
    sample_indices = [min(idx, total_frames - 1) for idx in sample_indices]

    sampled_frames = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sampled_frames.append((idx, frame.copy(), gray))

    cap.release()

    if len(sampled_frames) < 2:
        idx, frame, _ = sampled_frames[0]
        h, w = frame.shape[:2]
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), frame, w, h, idx

    flow_magnitudes = []
    for j in range(len(sampled_frames) - 1):
        gray_a = sampled_frames[j][2]
        gray_b = sampled_frames[j + 1][2]
        flow = cv2.calcOpticalFlowFarneback(
            gray_a, gray_b, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        flow_magnitudes.append(float(magnitude.mean()))

    window_size = min(window_size, len(flow_magnitudes))

    best_window_start = 0
    best_window_score = -1.0
    for start in range(len(flow_magnitudes) - window_size + 1):
        window_score = sum(flow_magnitudes[start:start + window_size])
        if window_score > best_window_score:
            best_window_score = window_score
            best_window_start = start

    candidate_start = best_window_start
    candidate_end = min(best_window_start + window_size + 1, len(sampled_frames))

    best_sharpness = -1.0
    best_idx = candidate_start
    for j in range(candidate_start, candidate_end):
        gray = sampled_frames[j][2]
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        if sharpness > best_sharpness:
            best_sharpness = sharpness
            best_idx = j

    chosen_frame_idx = sampled_frames[best_idx][0]
    chosen_frame_bgr = sampled_frames[best_idx][1]
    h, w = chosen_frame_bgr.shape[:2]

    peak_region_start = sampled_frames[candidate_start][0]
    peak_region_end = sampled_frames[candidate_end - 1][0]
    print(f"  Optical flow: peak motion in frames {peak_region_start}–{peak_region_end} "
          f"(of {total_frames})")
    print(f"  Selected frame {chosen_frame_idx} "
          f"(sharpness={best_sharpness:.1f}, ~{chosen_frame_idx / total_frames * 100:.0f}% through clip)")

    pil_image = Image.fromarray(cv2.cvtColor(chosen_frame_bgr, cv2.COLOR_BGR2RGB))
    return pil_image, chosen_frame_bgr, w, h, chosen_frame_idx


def run_cosmos_reasoner(model, processor, pil_image: Image.Image, prompt: str, max_new_tokens: int = 8192) -> str:
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text], images=[pil_image], return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = output_ids[0, input_len:]
    return processor.decode(generated, skip_special_tokens=True)


def extract_answer(text: str) -> Dict[str, Any]:
    if not text:
        return {"raw": text}
    start_idx = text.find("<answer>")
    end_idx = text.find("</answer>")
    if start_idx != -1 and end_idx != -1:
        json_str = text[start_idx + 8:end_idx].strip()
    else:
        json_str = text.strip()
    if json_str.startswith("```"):
        lines = json_str.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        json_str = "\n".join(lines).strip()
    try:
        repaired = repair_json(json_str, return_objects=True)
        if isinstance(repaired, dict) and len(repaired) > 0:
            return repaired
    except Exception as e:
        print(f"  [DEBUG] json_repair failed: {e}")
    return {"raw": text}


def extract_thinking(text: str) -> Optional[str]:
    if not text:
        return None
    start_idx = text.find("<think>")
    end_idx = text.find("</think>")
    if start_idx != -1 and end_idx != -1:
        return text[start_idx + 7:end_idx].strip()
    return None


def validate_variation_ontology(ontology: Dict[str, Any]) -> Tuple[bool, List[str]]:
    missing = []
    if not ontology.get("location"):
        missing.append("location")
    if not ontology.get("task"):
        missing.append("task")
    if not ontology.get("key_objects"):
        missing.append("key_objects")
    if not ontology.get("scene_variations"):
        missing.append("scene_variations")
    if not ontology.get("dataset_gaps"):
        missing.append("dataset_gaps")
    return len(missing) == 0, missing


def denormalize_bbox(bbox_norm: List[float], img_w: int, img_h: int) -> Dict[str, int]:
    x1_n, y1_n, x2_n, y2_n = bbox_norm
    return {
        "x1": int(x1_n * img_w / 1000),
        "y1": int(y1_n * img_h / 1000),
        "x2": int(x2_n * img_w / 1000),
        "y2": int(y2_n * img_h / 1000),
    }


def save_visual_qa(
    frame_bgr: np.ndarray,
    objects_with_bbox: List[Dict[str, Any]],
    clip_name: str,
    img_w: int,
    img_h: int,
    frame_idx: int,
    save_path: str,
) -> None:
    """Save a bbox-overlay image + a per-object crop grid to disk. Headless — never
    calls plt.show(), since batch runs happen on a server with no display attached."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    plot_objects = []
    for obj in objects_with_bbox:
        bbox = obj.get("bbox_2d")
        if not bbox or not isinstance(bbox, list) or len(bbox) != 4:
            continue
        px = denormalize_bbox(bbox, img_w, img_h)
        plot_objects.append({
            "name": obj.get("name", "unknown"),
            "rank": obj.get("rank", "?"),
            "norm": bbox,
            "px": px,
        })

    if not plot_objects:
        print("  ⚠️  No valid bounding boxes to save")
        return

    fig, (ax_clean, ax_ann) = plt.subplots(1, 2, figsize=(20, 8))

    ax_clean.imshow(frame_rgb)
    ax_clean.set_title("Selected Frame (optical flow + sharpness)", fontsize=12, fontweight="bold", pad=10)
    ax_clean.axis("off")

    ax_ann.imshow(frame_rgb)
    for i, obj in enumerate(plot_objects):
        px = obj["px"]
        color = BOX_COLORS[i % len(BOX_COLORS)]
        x1, y1 = px["x1"], px["y1"]
        w = px["x2"] - px["x1"]
        h = px["y2"] - px["y1"]
        ax_ann.add_patch(patches.Rectangle((x1, y1), w, h, linewidth=2.5, edgecolor=color, facecolor=color, alpha=0.12))
        ax_ann.add_patch(patches.Rectangle((x1, y1), w, h, linewidth=2.5, edgecolor=color, facecolor="none"))
        ax_ann.text(
            x1, max(y1 - 8, 12), f" {obj['rank']}. {obj['name']} ",
            fontsize=9, fontweight="bold", color="white", verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.9),
        )

    ax_ann.set_title(f"Cosmos Reason Detection — {len(plot_objects)} objects", fontsize=12, fontweight="bold", pad=10)
    ax_ann.axis("off")
    fig.suptitle(f"Visual QA: {clip_name}  |  Frame {frame_idx}  |  {img_w}×{img_h}px", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    n = len(plot_objects)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    fig2, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_2d(axes if n > 1 else np.array([axes]))

    for i, obj in enumerate(plot_objects):
        ax = axes[i // cols, i % cols]
        px = obj["px"]
        x1, y1 = max(0, px["x1"]), max(0, px["y1"])
        x2, y2 = min(img_w, px["x2"]), min(img_h, px["y2"])
        crop = frame_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            ax.text(0.5, 0.5, "empty crop", ha="center", va="center", transform=ax.transAxes, fontsize=10, color="red")
        else:
            ax.imshow(crop)
        color = BOX_COLORS[i % len(BOX_COLORS)]
        ax.set_title(f"{obj['rank']}. {obj['name']}", fontsize=10, fontweight="bold", color=color, pad=6)
        ax.set_xlabel(f"{obj['norm']}", fontsize=8, color="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)
    for i in range(n, rows * cols):
        axes[i // cols, i % cols].axis("off")

    fig2.suptitle("Per-Object Crops — verify each box targets the right region", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    crops_path = save_path.replace(".jpg", "_crops.jpg")
    plt.savefig(crops_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    print(f"  Saved: {save_path}")
    print(f"  Saved: {crops_path}")


def run_reason_on_clip(
    clip_path: str,
    output_dir: str,
    model,
    processor,
    save_visual_qa_images: bool = True,
    num_objects: int = DEFAULT_NUM_OBJECTS,
    max_new_tokens: int = 8192,
    frame_n_samples: int = 30,
    frame_window_size: int = 5,
) -> Dict[str, Any]:
    """
    Run Cosmos Reason on one clip, producing a variation ontology.

    Writes to output_dir:
        ontology.json     — the full result (schema described in this module's docstring)
        _thinking.txt      — the model's <think> block, for debugging
        _visual_qa.jpg      — bbox overlay (+ _visual_qa_crops.jpg per-object crops)

    Args:
        num_objects: how many objects to ask the model to identify.
        max_new_tokens: generation length cap passed to run_cosmos_reasoner.
        frame_n_samples / frame_window_size: passed to select_representative_frame.

    Returns:
        The ontology dict (same content written to ontology.json).
    """
    os.makedirs(output_dir, exist_ok=True)
    clip_name = os.path.splitext(os.path.basename(clip_path))[0]

    print(f"Processing {clip_name}...")

    pil_image, frame_bgr, img_w, img_h, frame_idx = select_representative_frame(
        clip_path, n_samples=frame_n_samples, window_size=frame_window_size
    )
    if pil_image is None:
        raise ValueError(f"Frame extraction failed for {clip_path}")
    print(f"  Frame: {img_w}x{img_h}")

    t0 = time.time()
    prompt = build_bbox_variation_prompt(num_objects)
    raw_response = run_cosmos_reasoner(model, processor, pil_image, prompt, max_new_tokens=max_new_tokens)
    inference_time = time.time() - t0
    print(f"  Inference: {inference_time:.1f}s")

    ontology = extract_answer(raw_response)
    is_valid, missing = validate_variation_ontology(ontology)

    key_objects = ontology.get("key_objects", [])
    if key_objects and frame_bgr is not None and save_visual_qa_images:
        vis_path = os.path.join(output_dir, "_visual_qa.jpg")
        save_visual_qa(frame_bgr, key_objects, clip_name, img_w, img_h, frame_idx, vis_path)

    for obj in key_objects:
        bbox = obj.get("bbox_2d")
        if bbox and isinstance(bbox, list) and len(bbox) == 4:
            obj["bbox_2d_pixels"] = denormalize_bbox(bbox, img_w, img_h)
            obj["bbox_2d_unit"] = [round(c / 1000, 4) for c in bbox]

    result = {
        "clip_path": clip_path,
        "clip_name": os.path.basename(clip_path),
        "frame_index": frame_idx,
        "frame_dimensions": {"width": img_w, "height": img_h},
        "frame_selection": "optical_flow_sharpness",
        "coordinate_system": {
            "type": "normalized_0_1000",
            "description": "[x1,y1,x2,y2] top-left to bottom-right, normalized 0-1000.",
            "conversion": "pixel = normalized * dimension / 1000",
        },
        "variation_ontology": ontology,
        "raw_response": raw_response,
        "inference_time_s": round(inference_time, 2),
        "valid": is_valid,
        "missing": missing,
    }

    task = (ontology.get("task") or {}).get("name", "N/A")
    room = (ontology.get("location") or {}).get("room_type", "N/A")
    n_obj = len(key_objects)
    n_bbox = sum(1 for o in key_objects if o.get("bbox_2d") and len(o.get("bbox_2d", [])) == 4)
    valid_str = "✓" if is_valid else "⚠"
    miss_str = f" | missing: {missing}" if missing else ""
    print(f"  {valid_str} Room: {room} | Task: {task} | Objects: {n_obj} | Bboxes: {n_bbox}{miss_str}")

    total_variations = 0
    for obj in key_objects:
        for v_list in (obj.get("variations") or {}).values():
            if isinstance(v_list, list):
                total_variations += len(v_list)
    print(f"  Total variations enumerated: {total_variations}")

    out_path = os.path.join(output_dir, "ontology.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    thinking = extract_thinking(raw_response)
    think_path = os.path.join(output_dir, "_thinking.txt")
    with open(think_path, "w") as f:
        f.write(f"Clip:       {clip_name}\n")
        f.write(f"Room:       {room}\n")
        f.write(f"Task:       {task}\n")
        f.write(f"Objects:    {n_obj} ({n_bbox} with bbox)\n")
        f.write(f"Variations: {total_variations}\n")
        f.write(f"Frame:      {frame_idx} (optical flow + sharpness)\n")
        f.write(f"Inference:  {inference_time:.1f}s\n")
        f.write("\n" + "=" * 60 + "\n\n")
        f.write(thinking if thinking else "NO <think> BLOCK FOUND\n\nFull response:\n" + raw_response[:3000])

    print(f"  ✅ Saved to {out_path}")
    return result
