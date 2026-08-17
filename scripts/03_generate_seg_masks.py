"""
Step 3: Generate segmentation masks using SAM2 + bounding boxes from ontology.

Usage:
    python scripts/03_generate_seg_masks.py \
        --video path/to/video.mp4 \
        --ontology path/to/ontology.json \
        --target "White surface" \
        --subtract "White plastic bucket" "Crumpled paper" "Human hand" \
        --frame-index 0 \
        --sam2-checkpoint path/to/sam2_hiera_large.pt \
        --output-dir path/to/seg_masks/
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_ontology
from src.segmentation import run_segmentation_pipeline


def main():
    parser = argparse.ArgumentParser(description="Generate segmentation masks")
    parser.add_argument("--video", "-v", required=True, help="Input video path")
    parser.add_argument("--ontology", "-i", required=True, help="Ontology JSON path")
    parser.add_argument("--target", "-t", required=True, help="Target object name")
    parser.add_argument("--subtract", "-s", nargs="*", default=[], help="Objects to subtract")
    parser.add_argument("--frame-index", "-f", type=int, default=0, help="Reference frame index")
    parser.add_argument("--sam2-checkpoint", required=True, help="SAM2 checkpoint path")
    parser.add_argument("--sam2-config", default="sam2_hiera_l.yaml", help="SAM2 config name")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory")
    parser.add_argument("--frames-dir", default=None, help="Scratch frames dir (default: <output-dir>/_scratch/frames)")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--num-points", type=int, default=10, help="Sample points per mask when seeding SAM2 video tracking")
    args = parser.parse_args()

    ontology = load_ontology(args.ontology)

    mask_path = run_segmentation_pipeline(
        video_path=args.video,
        ontology=ontology,
        target_name=args.target,
        subtract_names=args.subtract,
        bbox_frame_idx=args.frame_index,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
        output_dir=args.output_dir,
        frames_dir=args.frames_dir,
        device=args.device,
        num_points=args.num_points,
    )

    print(f"\n✅ Mask saved to: {mask_path}")


if __name__ == "__main__":
    main()
