"""Configuration loading and validation."""

import yaml
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict


# Base directory for all pipeline data (checkpoints, repos, scratch, outputs) when a
# more specific path isn't given. Overridable via the DATA_ROOT env var so this works
# unmodified whether it's pointed at a plain EBS-mounted directory or (later) something
# backed by S3 sync. Never defaults to a root-owned path like /content or /config.
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.expanduser("~/cosmos-data"))

# Single source of truth for the SAM2 checkpoint URL — both PipelineConfig (below) and
# src/checkpoints.py read this instead of each hardcoding their own copy.
SAM2_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"


@dataclass
class TransferWeights:
    edge: float = 0.5
    vis: float = 0.025
    depth: float = 0.2
    seg: float = 0.175

    def total(self) -> float:
        return self.edge + self.vis + self.depth + self.seg

    def validate(self):
        total = self.total()
        if total > 1.0:
            print(
                f"  ⚠ WARNING: Control weights sum to {total:.3f} (> 1.0). "
                f"Cosmos will normalize them down proportionally.\n"
                f"  Effective weights after normalization:\n"
                f"    edge:  {self.edge:.3f} → {self.edge / total:.3f}\n"
                f"    vis:   {self.vis:.3f} → {self.vis / total:.3f}\n"
                f"    depth: {self.depth:.3f} → {self.depth / total:.3f}\n"
                f"    seg:   {self.seg:.3f} → {self.seg / total:.3f}"
            )
        else:
            print(f"  ✅ Control weights sum to {total:.3f} (≤ 1.0, no normalization)")


@dataclass
class JobConfig:
    """Configuration for a single synthetic data generation job against one
    already-curated, already-annotated clip. Still used by the individual step
    scripts (01-04) and run_full_pipeline.py for one-off / manual runs.
    """

    # Input
    video_path: str
    variation_ontology_path: str
    bbox_frame_index: int = 0

    # Segmentation targets
    target_object: str = ""
    subtract_objects: List[str] = field(default_factory=list)

    # Transfer settings
    prompt: str = ""
    guidance: int = 4
    weights: TransferWeights = field(default_factory=TransferWeights)
    disable_guardrails: bool = True

    # Output
    output_dir: str = ""
    job_name: str = ""

    # Paths to external tools. Empty string means "use PipelineConfig's value" —
    # resolved via resolve_job_paths() below, not defaulted to a hardcoded path here.
    cosmos_transfer_repo: str = ""
    sam2_checkpoint: str = ""
    sam2_config: str = "sam2_hiera_l.yaml"
    sam2_num_points: int = 10

    # GPT-4 Mini prompt generation
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"


@dataclass
class BatchJobConfig:
    """Configuration for a batch source job: one or more raw, multi-minute videos
    that get split by Cosmos Curate, annotated by Cosmos Reason, and prompted by
    GPT-4 Mini — all unattended (phase 1) — producing a directory of per-clip state
    that scripts/run_interactive_review.py (phase 2) then walks interactively.

    No per-clip YAML files are created; each clip's state lives under
    output_dir/<clip_name>/ (ontology.json, prompts.json, _visual_qa/, seg_masks/,
    transfer_output/).
    """

    # Input: a single long video, or a directory containing several.
    input_video_path: str
    output_dir: str
    job_name: str = ""

    # Cosmos Curate splitting params
    curate_splitting_algorithm: str = "transnetv2"
    curate_threshold: float = 0.2
    curate_min_length_s: float = 2.0
    curate_max_length_s: float = 4.0
    curate_max_length_mode: str = "stride"
    curate_num_download_workers: int = 2
    curate_transcode_cpus: int = 2
    curate_clip_writer_workers: int = 2

    # Cosmos Reason params. Defaults match what src/reason.py has always done —
    # these just make it overridable per job instead of only editable in code.
    reason_model_id: str = "nvidia/Cosmos3-Nano"
    reason_num_objects: int = 10          # how many objects to ask the model to identify
    reason_max_new_tokens: int = 8192     # generation length cap
    reason_frame_samples: int = 30        # frames sampled when picking the representative frame
    reason_frame_window: int = 5          # sliding-window size for peak-motion detection

    # GPT-4 Mini prompt generation
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.7
    openai_max_tokens: int = 500
    max_variations_per_object: int = 3

    # SAM2 tracking
    sam2_num_points: int = 10             # sample points per mask when seeding video tracking

    # Transfer defaults (used to pre-fill specs in phase 2; guidance/weights can
    # still be overridden interactively)
    guidance: int = 4
    weights: TransferWeights = field(default_factory=TransferWeights)
    disable_guardrails: bool = True

    # Paths to external tools. Empty string means "use PipelineConfig's value".
    cosmos_transfer_repo: str = ""
    cosmos_curate_repo: str = ""
    sam2_checkpoint: str = ""
    sam2_config: str = "sam2_hiera_l.yaml"


@dataclass
class PipelineConfig:
    """Global pipeline configuration — the single source of truth for paths shared
    across every job (checkpoints, external tool repos, device). Per-job configs may
    override any of these; when they leave a field empty, resolve_job_paths() below
    fills it in from here.
    """

    # HuggingFace auth
    hf_token: str = ""

    # Checkpoint / model weights storage
    checkpoint_dir: str = field(default_factory=lambda: os.path.join(DATA_ROOT, "checkpoints"))

    # SAM2
    sam2_checkpoint_url: str = SAM2_CHECKPOINT_URL
    sam2_config: str = "sam2_hiera_l.yaml"

    # External tool repos
    cosmos_transfer_repo: str = field(default_factory=lambda: os.path.join(DATA_ROOT, "cosmos-transfer2.5"))
    cosmos_curate_repo: str = field(default_factory=lambda: os.path.join(DATA_ROOT, "cosmos-curate"))
    cosmos_venv_python: str = ""

    # Checkpoint download tuning (src/checkpoints.py). Defaults match what this
    # pipeline has always used — override if your connection needs more headroom.
    wget_timeout_s: int = 120       # large checkpoint file downloads
    hf_api_timeout_s: int = 30      # HuggingFace API metadata calls (file listing, revision lookup)
    predict_exclude_prefixes: List[str] = field(
        default_factory=lambda: ["auto/", "robot/", "base/pre-trained/", "base/post-trained/"]
    )
    transfer_exclude_prefixes: List[str] = field(default_factory=lambda: ["auto/", "robot/"])

    # Device
    device: str = "cuda"

    @property
    def sam2_checkpoint_path(self) -> str:
        return os.path.join(self.checkpoint_dir, "sam2_hiera_large.pt")


def load_job_config(yaml_path: str) -> JobConfig:
    """Load a job configuration from a YAML file."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    weights_data = data.pop("transfer_weights", data.pop("weights", {}))
    weights = TransferWeights(**weights_data) if weights_data else TransferWeights()

    config = JobConfig(weights=weights, **data)
    return config


def load_batch_job_config(yaml_path: str) -> BatchJobConfig:
    """Load a batch source job configuration from a YAML file."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    weights_data = data.pop("transfer_weights", data.pop("weights", {}))
    weights = TransferWeights(**weights_data) if weights_data else TransferWeights()

    config = BatchJobConfig(weights=weights, **data)
    return config


def load_pipeline_config(yaml_path: str) -> PipelineConfig:
    """Load the global pipeline configuration."""
    if not os.path.exists(yaml_path):
        return PipelineConfig()

    with open(yaml_path) as f:
        data = yaml.safe_load(f) or {}

    return PipelineConfig(**data)


def resolve_job_paths(job, pipeline: PipelineConfig) -> None:
    """Fill in a JobConfig/BatchJobConfig's empty tool-path fields from the global
    PipelineConfig, in place. Job-level values (if set) always win.
    """
    if not job.cosmos_transfer_repo:
        job.cosmos_transfer_repo = pipeline.cosmos_transfer_repo
    if hasattr(job, "cosmos_curate_repo") and not job.cosmos_curate_repo:
        job.cosmos_curate_repo = pipeline.cosmos_curate_repo
    if not job.sam2_checkpoint:
        job.sam2_checkpoint = pipeline.sam2_checkpoint_path
