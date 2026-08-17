# Cosmos Synthetic Data Pipeline

Generate synthetic training data for robotics by transforming real video clips using NVIDIA Cosmos Transfer 2.5.

Given raw, multi-minute recordings of a task being performed, this pipeline splits them into clips, identifies the objects in each one, changes specific objects (color, texture, material) while preserving the task motion, spatial layout, and untouched objects, and produces synthetic training video. This creates diverse training data from a handful of real-world recordings.

## Pipeline

```
Long clips → Cosmos Curate (split) → Cosmos Reason (bboxes + variations) → GPT-4 Mini (prompts)
                                                                                    │
                        ┌───────────────────────────────────────────────────────────┘
                        ▼
          [phase boundary: everything above runs unattended, everything below is interactive]
                        │
                        ▼
     CLI: pick target/protect objects per clip → SAM2 (seg masks) → Cosmos Transfer 2.5
```

1. **Cosmos Curate** splits raw multi-minute recordings into clips at the length/frame rate Cosmos Transfer expects
2. **Cosmos Reason** (`nvidia/Cosmos3-Nano`) identifies objects per clip and suggests plausible variations
3. **GPT-4 Mini** turns those suggestions into detailed Cosmos Transfer prompts
4. You pick, per clip, which object to change and which to protect, on the CLI
5. **SAM2** generates pixel-level segmentation masks from the picked bounding boxes (with subtraction to isolate bare surfaces)
6. **Cosmos Transfer 2.5** produces the final synthetic video using text prompt + 4 control maps (edge, depth, vis, seg)

Steps 1–3 run unattended across every clip (**phase 1**); steps 4–6 are a separate, resumable, interactive command you run per clip (**phase 2**) — see [docs/pipeline_overview.md](docs/pipeline_overview.md) for the full architecture and control map documentation.

## Quick Start

```bash
# Clone
git clone https://github.com/shreyj84/CosmosSynthethicDataGenerationCode.git
cd CosmosSynthethicDataGenerationCode

# Setup environment (installs Cosmos Transfer, Cosmos Curate, SAM2, dependencies)
chmod +x scripts/setup_environment.sh
./scripts/setup_environment.sh

# Set tokens
export HF_TOKEN=your_huggingface_token
export OPENAI_API_KEY=your_openai_key  # optional, for GPT-4 Mini

# Optional: point everything (checkpoints, repos, scratch) somewhere other than
# $HOME/cosmos-data — e.g. a larger EBS data volume
export DATA_ROOT=/data/cosmos

# Download checkpoints (Cosmos Predict/Transfer 2.5, SAM2, TransNetV2 — ~15 GB)
python scripts/download_checkpoints.py

# Phase 1 — unattended: split your raw clips, annotate them, generate candidate prompts
python scripts/run_batch_prepare.py --job config/jobs/bucket_cleaning_batch.yaml

# Phase 2 — interactive: pick objects per clip, run SAM2 + Cosmos Transfer
python scripts/run_interactive_review.py --job config/jobs/bucket_cleaning_batch.yaml
```

Phase 2 is safe to stop (Ctrl-C) and re-run — it only processes clips that haven't completed yet.

## Run Individual Steps

Useful for debugging one clip, or for hand-annotated single-clip jobs (see `JobConfig` in `src/config.py`) instead of the batch flow above.

```bash
# Step 1: Generate (or validate an existing) ontology for one clip
python scripts/01_run_cosmos_reason.py --generate --clip path/to/clip.mp4 --output-dir path/to/out/
python scripts/01_run_cosmos_reason.py --input path/to/out/ontology.json

# Step 2: Generate prompts
python scripts/02_generate_prompts.py \
    --input path/to/out/ontology.json \
    --target "White surface" \
    --variations "change color to blue" "change material to marble" \
    --output prompts.json

# Step 3: Generate segmentation masks
python scripts/03_generate_seg_masks.py \
    --video path/to/clip.mp4 \
    --ontology path/to/out/ontology.json \
    --target "White surface" \
    --subtract "White plastic bucket" "Crumpled paper" "Human hand" \
    --frame-index 0 \
    --sam2-checkpoint ${DATA_ROOT:-$HOME/cosmos-data}/checkpoints/sam2_hiera_large.pt \
    --output-dir seg_masks/

# Step 4: Run Cosmos Transfer
python scripts/04_run_cosmos_transfer.py \
    --prompts prompts.json \
    --video path/to/clip.mp4 \
    --seg-mask seg_masks/seg_mask.mp4 \
    --output-dir outputs/ \
    --cosmos-repo /path/to/cosmos-transfer2.5
```

Every step above also has optional tuning flags (`python scripts/01_run_cosmos_reason.py --help`, etc.) — none are required, they just override a built-in default. See [Tuning](#tuning-optional) below for what each one does and its default value.

## Job Configuration

**Batch jobs** (the normal path — raw multi-minute clips in, phase 1 + phase 2 as above) are defined in `config/jobs/*.yaml`:

```yaml
input_video_path: /data/raw_videos/bucket_cleaning/   # a long video, or a directory of them
output_dir: /data/cosmos_outputs/bucket_cleaning_batch
job_name: bucket_cleaning_batch

curate_splitting_algorithm: transnetv2
curate_threshold: 0.2
curate_min_length_s: 2.0
curate_max_length_s: 4.0
curate_max_length_mode: stride

reason_model_id: nvidia/Cosmos3-Nano

openai_model: gpt-4o-mini
max_variations_per_object: 3

guidance: 4
transfer_weights:
  edge: 0.5
  vis: 0.025
  depth: 0.2
  seg: 0.175

disable_guardrails: true
```

See [config/jobs/](config/jobs/) for full examples. Any tool path left unset here (`cosmos_transfer_repo`, `cosmos_curate_repo`, `sam2_checkpoint`) falls back to [config/default.yaml](config/default.yaml)'s global settings.

**Single-clip jobs** (skip Curate/Reason, everything pre-annotated by hand) use the older `JobConfig` schema — `video_path`, `variation_ontology_path`, `target_object`, `subtract_objects`, `prompt` — consumed by `scripts/run_full_pipeline.py` and the individual step scripts above.

## Tuning (optional)

Every one of these has a built-in default matching what the pipeline has always done — you never have to set any of them. They exist so you *can* override behavior without editing code. Add any of these keys to a job YAML (`config/jobs/*.yaml`) to override per job, or pass the matching flag to the individual step script for a one-off run (see `--help` on each script for the exact flag name).

| YAML key | Default | What it controls |
|---|---|---|
| `reason_num_objects` | `10` | How many objects Cosmos Reason is asked to identify per frame |
| `reason_max_new_tokens` | `8192` | Generation length cap for Cosmos Reason's response |
| `reason_frame_samples` | `30` | How many frames are sampled when picking the clip's representative frame |
| `reason_frame_window` | `5` | Sliding-window size used to find the frame of peak motion |
| `openai_temperature` | `0.7` | GPT-4 Mini sampling temperature when writing prompts |
| `openai_max_tokens` | `500` | Max length of each generated prompt |
| `sam2_num_points` | `10` | Sample points per object mask when seeding SAM2's video tracking |

`config/default.yaml` (global, not per-job) also has a few checkpoint-download knobs, off by default and only worth touching if downloads are slow or NVIDIA changes a repo's file layout — see [Known Issues](#known-issues) below.

## Control Weight Reference

| Weight | What it does | Range | Tip |
|--------|-------------|-------|-----|
| **edge** | Preserves object shapes/outlines | 0.4–0.7 | Lower for texture changes |
| **vis** | Preserves original colors | 0.0–0.1 | Set to 0 for color swaps |
| **depth** | Preserves spatial layout | 0.15–0.3 | Higher if objects overlap |
| **seg** | Directs changes to masked regions | 0.1–0.3 | Needs seg_mask.mp4 |

**Critical:** If weights sum > 1.0, Cosmos normalizes them proportionally. Keep total ≤ 1.0.

## Hardware Requirements

- **GPU:** A100 80GB (720p) or A10G 24GB (480p minimum)
- **VRAM:** 65 GB for 720p, ~24 GB for 480p with guardrails disabled. Cosmos Reason's checkpoint (~32GB) only needs to be loaded during phase 1 — it and SAM2/Transfer never share a process, so they don't compete for VRAM.
- **Disk:** 100 GB minimum (checkpoints + outputs) — more if `DATA_ROOT` also holds Cosmos Curate's raw/intermediate clips
- **Python:** 3.10 required for the Cosmos Transfer venv (flash-attn constraint) — see `scripts/setup_environment.sh`, which manages this in its own `.venv` separately from the orchestration layer's Python

See [docs/setup_aws.md](docs/setup_aws.md) for AWS-specific instructions.

## Known Issues

- **`nvidia/Cosmos3-Nano` needs a recent `transformers`/`diffusers`:** `Cosmos3OmniForConditionalGeneration` is a very new model class. `requirements.txt` installs `diffusers` from git main and leaves `transformers` unpinned for this reason — if loading fails with an import/class error, try the latest release of each.
- **GroundingDINO fails < 800px:** not used — this pipeline uses Cosmos Reason's bounding boxes instead, which don't have this limitation.
- **Cosmos Transfer is archived:** NVIDIA has moved to Cosmos 3. This pipeline works with Transfer 2.5.
- **InternVideo2 / Qwen2.5-VL (Cosmos Curate's embedding/captioning models) are intentionally not used** and not downloaded by this pipeline (`--no-generate-embeddings --no-generate-captions`) — see the project's separate notes on that decision.
- **Checkpoint downloads timing out on a slow connection:** `python scripts/download_checkpoints.py` uses `wget` with a 120s timeout per file and a 30s timeout for HuggingFace API lookups. If you're hitting timeouts, raise `wget_timeout_s` / `hf_api_timeout_s` in `config/default.yaml` — see [config/default.yaml](config/default.yaml) for the commented-out example.
- **`⚠ WARNING: ... matched none of them` during checkpoint download:** `download_checkpoints.py` filters each HuggingFace repo's file list down to what this pipeline actually needs (`predict_exclude_prefixes` / `transfer_exclude_prefixes` in `config/default.yaml`). This warning means NVIDIA restructured that repo's folders and the filter no longer matches anything — check the repo on HuggingFace and update the prefix list rather than assuming there's nothing to download.

## Project Structure

```
CosmosSynthethicDataGenerationCode/
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   ├── default.yaml                      # global config (paths, device)
│   └── jobs/                             # per-job configs
│       ├── bucket_cleaning_batch.yaml
│       ├── pillow_casing_batch.yaml
│       └── fixing_bed_batch.yaml
├── scripts/
│   ├── setup_environment.sh              # full environment setup (Transfer, Curate, SAM2)
│   ├── download_checkpoints.py           # checkpoint download/caching
│   ├── run_batch_prepare.py              # phase 1: curate -> reason -> prompts (unattended)
│   ├── run_interactive_review.py         # phase 2: CLI object picking -> SAM2 -> transfer
│   ├── run_full_pipeline.py              # single pre-annotated clip, all steps
│   ├── 01_run_cosmos_reason.py           # generate or validate one ontology
│   ├── 02_generate_prompts.py            # GPT-4 Mini -> prompts
│   ├── 03_generate_seg_masks.py          # SAM2 -> seg masks
│   └── 04_run_cosmos_transfer.py         # prompts + masks -> synthetic video
├── src/
│   ├── __init__.py
│   ├── config.py                         # config loading + validation (DATA_ROOT-based)
│   ├── utils.py                          # video I/O, frame extraction
│   ├── curate.py                         # Cosmos Curate wrapper (splitting)
│   ├── reason.py                         # Cosmos Reason wrapper (bboxes + variations)
│   ├── segmentation.py                   # SAM2 pipeline
│   ├── prompt_generator.py               # GPT-4 Mini integration
│   ├── transfer.py                       # Cosmos Transfer wrapper
│   └── checkpoints.py                    # checkpoint download/caching
├── docs/
│   ├── pipeline_overview.md              # architecture + control map documentation
│   └── setup_aws.md                      # AWS setup guide
└── tests/
    └── test_config.py                    # config validation + ontology schema tests
```
