"""Basic tests for config loading and weight validation."""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import inspect

from src.config import (
    TransferWeights,
    JobConfig,
    BatchJobConfig,
    PipelineConfig,
    DATA_ROOT,
    SAM2_CHECKPOINT_URL,
    load_job_config,
    load_batch_job_config,
    load_pipeline_config,
    resolve_job_paths,
)
from src.utils import get_objects_by_name
from src.checkpoints import download_all_checkpoints, DEFAULT_WGET_TIMEOUT_S, DEFAULT_HF_API_TIMEOUT_S


def test_weight_total():
    w = TransferWeights(edge=0.5, vis=0.025, depth=0.2, seg=0.175)
    assert w.total() == pytest.approx(0.9)


def test_weight_normalization_warning(capsys):
    w = TransferWeights(edge=1.0, vis=0.1, depth=0.8, seg=0.3)
    w.validate()
    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "normalize" in captured.out.lower()


def test_weight_no_warning(capsys):
    w = TransferWeights(edge=0.5, vis=0.025, depth=0.2, seg=0.175)
    w.validate()
    captured = capsys.readouterr()
    assert "✅" in captured.out


def test_default_weights():
    w = TransferWeights()
    assert w.total() <= 1.0


def test_load_job_config(tmp_path):
    config_file = tmp_path / "test_job.yaml"
    config_file.write_text("""
video_path: /data/test.mp4
variation_ontology_path: /data/test.json
target_object: "table"
subtract_objects:
  - "cup"
  - "hand"
transfer_weights:
  edge: 0.6
  vis: 0.0
  depth: 0.2
  seg: 0.2
output_dir: /data/output
job_name: test_job
""")

    config = load_job_config(str(config_file))
    assert config.video_path == "/data/test.mp4"
    assert config.target_object == "table"
    assert len(config.subtract_objects) == 2
    assert config.weights.edge == 0.6
    assert config.weights.total() == pytest.approx(1.0)


def test_load_batch_job_config(tmp_path):
    config_file = tmp_path / "test_batch.yaml"
    config_file.write_text("""
input_video_path: /data/raw_videos/task/
output_dir: /data/output/task_batch
curate_max_length_s: 5.0
max_variations_per_object: 2
transfer_weights:
  edge: 0.5
  vis: 0.025
  depth: 0.2
  seg: 0.175
""")

    config = load_batch_job_config(str(config_file))
    assert config.input_video_path == "/data/raw_videos/task/"
    assert config.output_dir == "/data/output/task_batch"
    assert config.curate_max_length_s == 5.0
    assert config.max_variations_per_object == 2
    assert config.weights.total() == pytest.approx(0.9)
    # empty until resolved against a PipelineConfig
    assert config.cosmos_transfer_repo == ""
    assert config.cosmos_curate_repo == ""


def test_pipeline_config_defaults_are_not_colab_paths():
    pipeline = PipelineConfig()
    assert not pipeline.checkpoint_dir.startswith("/content")
    assert not pipeline.cosmos_transfer_repo.startswith("/content")
    assert not pipeline.cosmos_curate_repo.startswith("/content")
    assert pipeline.checkpoint_dir.startswith(DATA_ROOT)
    assert pipeline.sam2_checkpoint_path == os.path.join(pipeline.checkpoint_dir, "sam2_hiera_large.pt")


def test_load_pipeline_config_missing_file_falls_back_to_defaults():
    pipeline = load_pipeline_config("/no/such/file.yaml")
    assert pipeline == PipelineConfig()


def test_resolve_job_paths_fills_in_blanks_but_job_overrides_win():
    pipeline = PipelineConfig(
        checkpoint_dir="/data/checkpoints",
        cosmos_transfer_repo="/data/cosmos-transfer2.5",
        cosmos_curate_repo="/data/cosmos-curate",
    )

    blank_job = BatchJobConfig(input_video_path="/data/in", output_dir="/data/out")
    resolve_job_paths(blank_job, pipeline)
    assert blank_job.cosmos_transfer_repo == "/data/cosmos-transfer2.5"
    assert blank_job.cosmos_curate_repo == "/data/cosmos-curate"
    assert blank_job.sam2_checkpoint == os.path.join("/data/checkpoints", "sam2_hiera_large.pt")

    overridden_job = BatchJobConfig(
        input_video_path="/data/in",
        output_dir="/data/out",
        cosmos_transfer_repo="/custom/cosmos-transfer2.5",
    )
    resolve_job_paths(overridden_job, pipeline)
    assert overridden_job.cosmos_transfer_repo == "/custom/cosmos-transfer2.5"
    assert overridden_job.cosmos_curate_repo == "/data/cosmos-curate"


def _sample_reason_shaped_ontology():
    """Shaped exactly like src/reason.py:run_reason_on_clip()'s output — locks in
    that Cosmos Reason's ontology is a drop-in replacement for a hand-produced one."""
    return {
        "clip_path": "/data/clips/bucket_cleaning_001.mp4",
        "frame_index": 12,
        "frame_dimensions": {"width": 1280, "height": 720},
        "variation_ontology": {
            "location": {"room_type": "kitchen", "setting": "indoor", "lighting_condition": "bright"},
            "task": {"name": "bucket cleaning", "description": "wiping a countertop"},
            "key_objects": [
                {
                    "rank": 1,
                    "name": "White surface",
                    "bbox_2d": [100, 200, 400, 500],
                    "bbox_2d_pixels": {"x1": 128, "y1": 144, "x2": 512, "y2": 360},
                    "current_appearance": {
                        "color_description": "white", "material": "laminate", "texture": "smooth",
                    },
                    "variations": {
                        "color_and_appearance": ["blue", "green", "red"],
                        "material_and_texture": ["marble", "wood grain"],
                    },
                },
                {
                    "rank": 2,
                    "name": "White plastic bucket",
                    "bbox_2d": [500, 300, 700, 600],
                    "bbox_2d_pixels": {"x1": 640, "y1": 216, "x2": 896, "y2": 432},
                    "current_appearance": {"color_description": "white", "material": "plastic", "texture": "smooth"},
                    "variations": {"color_and_appearance": [], "material_and_texture": []},
                },
            ],
            "scene_variations": {},
            "dataset_gaps": {},
        },
    }


def test_reason_ontology_schema_matches_get_objects_by_name():
    result = _sample_reason_shaped_ontology()
    objects = get_objects_by_name(result)
    assert set(objects.keys()) == {"White surface", "White plastic bucket"}
    assert objects["White surface"] == {"x1": 128, "y1": 144, "x2": 512, "y2": 360}


def test_sam2_checkpoint_url_is_defined_once():
    """PipelineConfig and src/checkpoints.py must read the same URL, not each keep
    their own copy — this was flagged in the hardcoded-values audit and fixed by
    centralizing on config.SAM2_CHECKPOINT_URL."""
    assert PipelineConfig().sam2_checkpoint_url == SAM2_CHECKPOINT_URL
    download_defaults = inspect.signature(download_all_checkpoints).parameters
    assert download_defaults["sam2_checkpoint_url"].default == SAM2_CHECKPOINT_URL


def test_checkpoint_download_timeouts_are_parameterized_with_stable_defaults():
    """Timeouts used to be inline string literals scattered across checkpoints.py
    ("--timeout=120", "--timeout=30" in different places) with no way to override —
    now they're named parameters. Defaults must stay what they've always been."""
    download_defaults = inspect.signature(download_all_checkpoints).parameters
    assert download_defaults["timeout_s"].default == DEFAULT_WGET_TIMEOUT_S == 120
    assert download_defaults["api_timeout_s"].default == DEFAULT_HF_API_TIMEOUT_S == 30
    pipeline = PipelineConfig()
    assert pipeline.wget_timeout_s == 120
    assert pipeline.hf_api_timeout_s == 30


def test_batch_job_config_tuning_knobs_default_to_prior_hardcoded_behavior():
    """New optional fields must default to exactly what was hardcoded before —
    parameterizing these shouldn't silently change pipeline behavior."""
    job = BatchJobConfig(input_video_path="/data/in", output_dir="/data/out")
    assert job.reason_num_objects == 10  # the prompt always asked for "up to 10"
    assert job.reason_max_new_tokens == 8192
    assert job.reason_frame_samples == 30
    assert job.reason_frame_window == 5
    assert job.openai_temperature == 0.7
    assert job.openai_max_tokens == 500
    assert job.sam2_num_points == 10
