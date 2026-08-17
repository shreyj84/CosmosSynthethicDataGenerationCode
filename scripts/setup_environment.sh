#!/bin/bash
# Setup script for Cosmos Synthetic Data Pipeline (AWS Ubuntu 22/24; also works on a
# root-shell environment like Colab, though this repo no longer targets Colab).
#
# Usage:
#   chmod +x scripts/setup_environment.sh
#   ./scripts/setup_environment.sh
#
# Override any of these before running if the defaults don't fit your box:
#   DATA_ROOT             base dir for repos/checkpoints/scratch (default: $HOME/cosmos-data)
#   COSMOS_TRANSFER_REPO  (default: $DATA_ROOT/cosmos-transfer2.5)
#   COSMOS_CURATE_REPO    (default: $DATA_ROOT/cosmos-curate)
#   SAM2_REPO             (default: $DATA_ROOT/Grounded-SAM-2)

set -e

echo "============================================"
echo "Cosmos Synthetic Data — Environment Setup"
echo "============================================"

DATA_ROOT="${DATA_ROOT:-$HOME/cosmos-data}"
mkdir -p "$DATA_ROOT"

# Root (Colab-style) shells don't need sudo and often don't have it installed;
# a normal AWS EC2 user does.
if [ "$(id -u)" -eq 0 ]; then
    SUDO=""
else
    SUDO="sudo"
fi

# System deps
echo "[1/6] Installing system dependencies..."
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq curl ffmpeg libx11-dev tree wget > /dev/null

# Install uv
echo "[2/6] Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Clone and setup Cosmos Transfer 2.5
echo "[3/6] Setting up Cosmos Transfer 2.5..."
COSMOS_DIR="${COSMOS_TRANSFER_REPO:-$DATA_ROOT/cosmos-transfer2.5}"

if [ ! -d "$COSMOS_DIR" ]; then
    git clone https://github.com/nvidia-cosmos/cosmos-transfer2.5.git "$COSMOS_DIR"
fi

cd "$COSMOS_DIR"

# CRITICAL: Force Python 3.10 — flash-attn only has cp310 wheels
echo "3.10" > .python-version
rm -rf .venv
uv venv .venv --python 3.10 --seed
uv sync --extra=cu128

echo "Verifying Cosmos Transfer install..."
.venv/bin/python --version
.venv/bin/python -c "import flash_attn; import cosmos_oss; print('Cosmos Transfer: OK')"

# Clone and setup Cosmos Curate (splits long source videos into Transfer-ready clips)
echo "[4/6] Setting up Cosmos Curate..."
CURATE_DIR="${COSMOS_CURATE_REPO:-$DATA_ROOT/cosmos-curate}"

if [ ! -d "$CURATE_DIR" ]; then
    git clone https://github.com/nvidia-cosmos/cosmos-curate.git "$CURATE_DIR"
fi

cd "$CURATE_DIR"
curl -fsSL https://pixi.sh/install.sh | bash
export PATH="$HOME/.pixi/bin:$PATH"
pixi install
pixi run pip install -e . --no-deps
# Extra deps the split pipeline needs beyond its own pyproject (carried over from a
# working setup — not independently version-audited).
pixi run pip install \
  "transformers==4.49.0" \
  "tokenizers==0.21.0" \
  "ray[data,default,serve]==2.55.0" \
  einops \
  decord \
  opencv-python \
  ftfy \
  sentencepiece \
  accelerate \
  torchvision \
  pillow \
  scipy \
  scikit-learn \
  imageio \
  av

echo "Verifying Cosmos Curate install..."
pixi run python -c "import cosmos_curate; print('Cosmos Curate: OK')"

# Clone and setup Grounded-SAM-2
echo "[5/6] Setting up SAM2..."
SAM2_DIR="${SAM2_REPO:-$DATA_ROOT/Grounded-SAM-2}"

if [ ! -d "$SAM2_DIR" ]; then
    cd "$DATA_ROOT"
    git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
fi

cd "$SAM2_DIR"
pip install -e . --quiet
pip install transformers supervision --quiet

echo "Verifying SAM2 install..."
python -c "from sam2.build_sam import build_sam2; print('SAM2: OK')"

# Install pipeline (orchestration-layer) dependencies, including Cosmos Reason's.
# NOTE: nvidia/Cosmos3-Nano needs a very recent transformers (for
# Cosmos3OmniForConditionalGeneration) and diffusers built from git main — neither is
# pinned in requirements.txt because they weren't pinned in the working Colab
# environment this was ported from either. If model loading fails with an unknown
# class/import error, check for a newer transformers release or reinstall from git
# main the same way diffusers is here.
echo "[6/6] Installing pipeline dependencies..."
pip install -r "$(dirname "$0")/../requirements.txt" --quiet

echo ""
echo "============================================"
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Set HF_TOKEN:    export HF_TOKEN=your_token"
echo "  2. Login to HF:     huggingface-cli login --token \$HF_TOKEN"
echo "  3. Download models:  python scripts/download_checkpoints.py"
echo "  4. Prepare clips:    python scripts/run_batch_prepare.py --job config/jobs/bucket_cleaning_batch.yaml"
echo "  5. Review + render:  python scripts/run_interactive_review.py --job config/jobs/bucket_cleaning_batch.yaml"
echo "============================================"
