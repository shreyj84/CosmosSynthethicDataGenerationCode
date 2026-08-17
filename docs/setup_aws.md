# AWS Setup Guide

## Recommended Instance

| Instance | GPU | VRAM | Resolution | Cost |
|----------|-----|------|------------|------|
| p4d.24xlarge | A100 80GB | 80 GB | 720p | ~$32/hr |
| g5.2xlarge | A10G 24GB | 24 GB | 480p only | ~$1.2/hr |
| p5.48xlarge | H100 80GB | 80 GB | 720p (fastest) | ~$98/hr |

For experimentation, the g5.2xlarge at 480p is cost-effective.
For production runs, the p4d.24xlarge gives full 720p quality.

Cosmos Reason's checkpoint (~32GB) only loads during phase 1 (`run_batch_prepare.py`);
SAM2 and Cosmos Transfer only load during phase 2 (`run_interactive_review.py`) — they
never run in the same process, so they don't add to each other's VRAM requirement.

## AMI

Use the **Deep Learning AMI (Ubuntu 22.04)** with CUDA 12.x pre-installed.

## Storage (EBS)

- **Root volume:** 200 GB minimum (for repos, checkpoints, venvs)
- **Data volume:** a separate EBS volume, sized for your video library + outputs.
  Don't rely on the root volume for this.
- Checkpoints alone are ~15 GB (not counting Cosmos Reason's own ~32GB, which caches
  under `~/.cache/huggingface` the first time it's used)

This pipeline never talks to AWS APIs directly (no boto3/S3 calls anywhere) — every
path is a plain local filesystem path, so an EBS volume just needs to be attached and
mounted like any other disk:

```bash
# 1. Attach the EBS volume to the instance (console, or aws ec2 attach-volume), then
#    find its device name
lsblk

# 2. Format it (first time only — skip if it already has a filesystem)
sudo mkfs -t ext4 /dev/nvme1n1        # device name from lsblk; nvme1n1 is typical on newer instance types

# 3. Mount it
sudo mkdir -p /data
sudo mount /dev/nvme1n1 /data
sudo chown $USER:$USER /data

# 4. Make it survive a reboot
echo "$(sudo blkid -s UUID -o value /dev/nvme1n1) /data ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab
```

Point `DATA_ROOT` at the data volume's mount point so checkpoints, cloned repos, and
scratch/output files all land there instead of the root volume:

```bash
export DATA_ROOT=/data/cosmos
```

(`src/config.py`'s `PipelineConfig` derives every default path — `checkpoint_dir`,
`cosmos_transfer_repo`, `cosmos_curate_repo` — from `DATA_ROOT`. Job configs'
`input_video_path`/`output_dir` are independent of this and can point anywhere.)

## Setup Steps

```bash
# 1. SSH into instance
ssh -i your-key.pem ubuntu@your-instance-ip

# 2. Clone the pipeline repo
git clone https://github.com/shreyj84/CosmosSynthethicDataGenerationCode.git
cd CosmosSynthethicDataGenerationCode

# 3. Set tokens and DATA_ROOT
export HF_TOKEN=your_huggingface_token
export OPENAI_API_KEY=your_openai_key  # for GPT-4 Mini prompt generation
export DATA_ROOT=/data/cosmos          # point at your mounted EBS data volume

# 4. Run setup (installs Cosmos Transfer, Cosmos Curate, SAM2 — each in their own
#    venv/pixi environment; uses sudo automatically since the ubuntu user isn't root)
chmod +x scripts/setup_environment.sh
./scripts/setup_environment.sh

# 5. Download checkpoints (one-time)
python scripts/download_checkpoints.py

# 6. Get your raw video files onto the instance — scp, aws s3 cp/sync, or mount the
#    data volume if it already has them. Point a job config's input_video_path at
#    wherever they land (see config/jobs/*.yaml).

# 7. Run a job
python scripts/run_batch_prepare.py --job config/jobs/bucket_cleaning_batch.yaml
python scripts/run_interactive_review.py --job config/jobs/bucket_cleaning_batch.yaml
```

### If checkpoint downloads are timing out

`download_checkpoints.py` uses `wget` under the hood with a 120s timeout per file and
a 30s timeout for HuggingFace API metadata calls — generous for typical AWS bandwidth,
but if your instance is on a constrained network, raise them in `config/default.yaml`:

```yaml
wget_timeout_s: 300
hf_api_timeout_s: 60
```

## Persistent Checkpoint Storage (EBS)

Save checkpoints to a persistent EBS volume so you don't re-download on new instances:

```bash
# Save after first download
python scripts/download_checkpoints.py --save-to /data/persistent/model_weights

# Restore on new instance
python scripts/download_checkpoints.py --restore-from /data/persistent/model_weights
```

## Key Differences from Colab (historical — this pipeline no longer targets Colab)

- No Google Drive — paths are just local filesystem paths under `DATA_ROOT`, backed
  by whatever EBS volume(s) you've mounted. If your video library instead lives in S3,
  sync it down (`aws s3 sync`) to a local path first — this pipeline doesn't call S3
  APIs directly.
- No `userdata.get()` — tokens come from environment variables (`HF_TOKEN`,
  `OPENAI_API_KEY`) or `config/default.yaml`'s `hf_token` field.
- Python 3.10 is still required for the Cosmos Transfer venv (flash-attn constraint) —
  `setup_environment.sh` manages this in its own `.venv`, separate from whatever
  Python runs the orchestration scripts.
- `setup_environment.sh` uses `sudo` for system package installs (skipped
  automatically if already running as root) — the default Colab root shell never
  needed this, but a normal AWS EC2 user does.
- No runtime disconnects — jobs run to completion. Phase 2
  (`run_interactive_review.py`) is still built to be resumable regardless, since it's
  an interactive session you may want to pause and come back to.
