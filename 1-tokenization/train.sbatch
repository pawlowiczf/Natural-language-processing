#!/bin/bash
#SBATCH --job-name=transformer-train
#SBATCH --output=train-%x-%A_%a.out
#SBATCH --error=train-%x-%A_%a.err
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

#SBATCH --account=plgar2026-gpu
#SBATCH --partition=plgrid-gpu-v100

source ~/.bashrc
conda activate plgpawlowiczfenv

cd $SLURM_SUBMIT_DIR

export HF_HUB_OFFLINE=1

TOKENIZERS=(bpe whitespace sentencepiece)
TOKENIZER=${TOKENIZERS[$SLURM_ARRAY_TASK_ID]}
echo "tokenizer: $TOKENIZER"

nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# Extra arguments given to sbatch after the script name are passed on, e.g. --max-steps 20
srun python train.py --tokenizer $TOKENIZER "$@"
