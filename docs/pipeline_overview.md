# Pipeline Overview

## What This Does

Generates synthetic training data for robotics by transforming real video clips.
Given raw, multi-minute recordings of a robot (or person) performing a task, the
pipeline splits them into clips, changes specific objects in the scene (color,
texture, material) while preserving the task, motion, and untouched objects.

## Architecture

```
┌───────────────────┐
│    Long Clips      │
│  (real footage,     │
│   multi-minute)     │
└─────────┬──────────┘
          ▼
┌───────────────────┐
│   Cosmos Curate     │   split to Transfer's recommended clip length/fps
│  (splitting)         │
└─────────┬──────────┘
          ▼  (one iteration per produced clip)
┌───────────────────┐     ┌──────────────────────┐
│   Cosmos Reason     │────▶│  Variation Ontology   │
│  (Cosmos3-Nano,      │     │  (bboxes + variations)│
│   object detect)     │     └──────────┬───────────┘
└───────────────────┘                  │
   ══════════════════ phase boundary ══│══════════════════════════
   phase 1: unattended                 │   phase 2: interactive
   ══════════════════════════════════════│══════════════════════════
                    ┌──────────────┐      │
                    │  GPT-4 Mini   │◀─────┘
                    │  (candidate   │
                    │   prompts)    │
                    └──────┬───────┘
                           ▼
              ┌─────────────────────────┐
              │   CLI: pick target +      │   you choose, per clip
              │   protect objects,         │
              │   pick/edit a prompt       │
              └───────────┬─────────────┘
                    ┌──────┴───────┐
                    ▼              ▼
             ┌────────────┐ ┌────────────┐
             │   SAM2      │ │  Auto-gen   │
             │ (seg masks) │ │ (edge, depth,│
             │             │ │  vis maps)   │
             └─────┬──────┘ └─────┬──────┘
                   │              │
                   ▼              ▼
             ┌─────────────────────────────────────┐
             │        Cosmos Transfer 2.5           │
             │  (text-guided video-to-video model)  │
             └───────────────┬─────────────────────┘
                             ▼
                    ┌─────────────────┐
                    │  Synthetic Video │
                    │  (augmented data)│
                    └─────────────────┘
```

## The Steps

### Step 1: Cosmos Curate → Clips
- **Input:** One or more raw, multi-minute videos
- **Output:** Short clips at the length/frame rate Cosmos Transfer expects
- **Script:** `src/curate.py`, invoked by `run_batch_prepare.py`

### Step 2: Cosmos Reason → Variation Ontology
- **Input:** One representative frame per clip (chosen by optical flow + sharpness)
- **Output:** JSON with bounding boxes for every detected object + suggested variations
- **Model:** `nvidia/Cosmos3-Nano`
- **Script:** `src/reason.py`, invoked by `run_batch_prepare.py`; `01_run_cosmos_reason.py --generate` for a single clip, or `--input` to validate/summarize an existing ontology JSON
- **Tunable (all optional, defaults shown):** how many objects it looks for
  (`reason_num_objects`, default 10), how long its response can be
  (`reason_max_new_tokens`, default 8192), and how the representative frame gets
  picked — frames sampled (`reason_frame_samples`, default 30) and the sliding-window
  size used to find peak motion (`reason_frame_window`, default 5). Set any of these in
  a job YAML or pass the matching `01_run_cosmos_reason.py` flag.

### Step 3: GPT-4 Mini → Candidate Transfer Prompts
- **Input:** Variation ontology + a suggested variation description (from the ontology itself)
- **Output:** Detailed text prompt for Cosmos Transfer, one candidate per suggested variation per object
- **Script:** `02_generate_prompts.py` (single clip); `run_batch_prepare.py` runs this across every clip's every detected object
- **Tunable (optional):** sampling temperature (`openai_temperature`, default 0.7) and
  max prompt length (`openai_max_tokens`, default 500)

**— phase boundary — everything above is unattended; everything below is interactive —**

### Step 4: Interactive object picking
- **Input:** A clip's ontology (from step 2) + candidate prompts (from step 3)
- **Output:** Which object to change, which objects to protect, and which prompt to use for this clip
- **Script:** `run_interactive_review.py` — numbered CLI list, no GUI. Resumable: a clip already fully processed is skipped on the next run.

### Step 5: SAM2 → Segmentation Masks
- **Input:** Video + the target/protect bounding boxes picked in step 4
- **Output:** `seg_mask.mp4` (binary mask video) + `seg_overlay.mp4` (visualization)
- **Key feature:** Subtraction — detects target object AND objects on top of it,
  then subtracts to get only the bare surface
- **Script:** `03_generate_seg_masks.py` / `src/segmentation.py`, invoked by `run_interactive_review.py`
- **Tunable (optional):** sample points per mask when seeding SAM2's video tracking
  (`sam2_num_points`, default 10) — more points can help tracking stick to a fiddly
  or thin object, at some extra compute cost

### Step 6: Cosmos Transfer → Synthetic Video
- **Input:** Original video + prompt + seg mask + auto-generated control maps
- **Output:** Transformed video with the specified changes
- **Script:** `04_run_cosmos_transfer.py` / `src/transfer.py`, invoked by `run_interactive_review.py`

## Control Maps Explained

Cosmos Transfer uses 4 control maps to guide generation:

| Control | What it preserves | Source | Typical weight |
|---------|-------------------|--------|----------------|
| **Edge** | Object shapes, outlines, silhouettes | Auto: Canny edge detection | 0.4–0.7 |
| **Depth** | 3D spatial layout, foreground/background | Auto: Video Depth Anything | 0.15–0.3 |
| **Vis** | Original color palette, lighting | Auto: Gaussian blur of input | 0.0–0.1 |
| **Seg** | Semantic regions (what to change vs protect) | Pre-computed: SAM2 pipeline | 0.1–0.3 |

### Weight Normalization Warning

**If weights sum to > 1.0, Cosmos normalizes them proportionally.**

Example: edge=1.0, depth=0.8 sums to 1.8.
Effective weights: edge=0.56, depth=0.44.

**Always keep weights ≤ 1.0 total to avoid surprises.**

## Prompt Writing Tips

1. **Describe the scene as it should look AFTER transformation** — not "change X to Y"
2. **Repeat the target color/material 2-3 times** in different contexts
3. **Explicitly state what stays the same** — "The bucket and paper towels retain
   their original appearance"
4. **Be specific about materials** — "smooth matte blue finish" not just "blue"
5. **Keep under 300 words** — the model has a token limit
6. **For texture changes, lower edge weight** — edge at 0.5+ preserves original
   texture, fighting the prompt

GPT-4 Mini (`src/prompt_generator.py`) already follows these rules when it turns a
Cosmos Reason variation suggestion into a candidate prompt — this list mainly matters
when writing one by hand during phase 2's interactive review.

## Segmentation Tips

1. **Subtraction approach:** Detect the surface AND objects on it, subtract objects
   from the surface mask to get bare surfaces only
2. **Frame index matters:** Bounding boxes come from whichever frame Cosmos Reason
   selected (optical flow + sharpness — see `src/reason.py:select_representative_frame`),
   recorded as `frame_index` in that clip's `ontology.json`. SAM2 tracking propagates
   both forward and backward from that frame.
3. **At < 800px resolution,** GroundingDINO struggles. This pipeline uses Cosmos
   Reason's bounding boxes instead, which don't have this limitation.
4. **Always inspect `_visual_qa.jpg`** (saved next to each clip's `ontology.json`)
   before picking objects in phase 2 — it shows every detected bbox overlaid on the
   selected frame, so you can sanity-check them before committing to a target.
