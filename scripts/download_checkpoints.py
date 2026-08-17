"""
Download all model checkpoints (Cosmos Predict/Transfer 2.5, SAM2, Video Depth
Anything, TransNetV2). See src/checkpoints.py for why some use wget and some use the
native HuggingFace client.

Does NOT download nvidia/Cosmos3-Nano (Cosmos Reason) — see src/reason.py, it caches
itself on first use. Does NOT download InternVideo2 or Qwen2.5-VL — this pipeline
doesn't use Cosmos Curate's embeddings/captioning features.

Usage:
    export HF_TOKEN=your_token
    python scripts/download_checkpoints.py

    # With persistent cache (EBS / S3-mounted path / etc.):
    python scripts/download_checkpoints.py --save-to /data/persistent/model_weights
    python scripts/download_checkpoints.py --restore-from /data/persistent/model_weights
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.checkpoints import (
    download_all_checkpoints,
    save_to_persistent_storage,
    restore_from_persistent_storage,
    link_to_hf_cache,
)
from src.config import load_pipeline_config


def main():
    parser = argparse.ArgumentParser(description="Download model checkpoints")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Local directory for checkpoints (default: pipeline config's checkpoint_dir)")
    parser.add_argument("--pipeline-config", default="config/default.yaml", help="Global pipeline config YAML path")
    parser.add_argument("--save-to", help="Save checkpoints to persistent storage after download")
    parser.add_argument("--restore-from", help="Restore from persistent storage instead of downloading")
    args = parser.parse_args()

    pipeline = load_pipeline_config(args.pipeline_config)
    checkpoint_dir = args.checkpoint_dir or pipeline.checkpoint_dir

    hf_token = pipeline.hf_token or os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("❌ Set HF_TOKEN environment variable (or hf_token in the pipeline config) first:")
        print("   export HF_TOKEN=your_huggingface_token")
        sys.exit(1)

    if args.restore_from:
        restore_from_persistent_storage(args.restore_from, checkpoint_dir)
        # Still need to link
        predict_dir = os.path.join(checkpoint_dir, "nvidia--Cosmos-Predict2.5-2B")
        transfer_dir = os.path.join(checkpoint_dir, "nvidia--Cosmos-Transfer2.5-2B")
        link_to_hf_cache("nvidia/Cosmos-Predict2.5-2B", predict_dir, hf_token, api_timeout_s=pipeline.hf_api_timeout_s)
        link_to_hf_cache("nvidia/Cosmos-Transfer2.5-2B", transfer_dir, hf_token, api_timeout_s=pipeline.hf_api_timeout_s)
    else:
        download_all_checkpoints(
            hf_token,
            checkpoint_dir,
            sam2_checkpoint_url=pipeline.sam2_checkpoint_url,
            predict_exclude_prefixes=pipeline.predict_exclude_prefixes,
            transfer_exclude_prefixes=pipeline.transfer_exclude_prefixes,
            timeout_s=pipeline.wget_timeout_s,
            api_timeout_s=pipeline.hf_api_timeout_s,
        )

        if args.save_to:
            save_to_persistent_storage(checkpoint_dir, args.save_to)

    print("\n✅ Checkpoints ready.")


if __name__ == "__main__":
    main()
