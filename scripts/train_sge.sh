#!/usr/bin/env bash
#$ -N dmmd_cifar10
#$ -P aihub_ucl
#$ -cwd
#$ -V
#$ -l gpu=true,gpu_type=h100
#$ -pe gpu 1
#$ -l tmem=16G
#$ -l h_rt=24:00:00
#$ -R y
#$ -j y
#$ -o .
#$ -notify

set -eo pipefail

# Submit from the experiment folder, or set DMMD_PROJECT_DIR explicitly.
PROJECT_DIR=${DMMD_PROJECT_DIR:-${SGE_O_WORKDIR:-$PWD}}
cd "$PROJECT_DIR"
if [[ ! -f train.py ]]; then
  echo "Run qsub from the dmmd_cifar10 folder, or set DMMD_PROJECT_DIR." >&2
  exit 1
fi

# Uses the environment active when submitting with qsub -V.
# Optionally export DMMD_CONDA_ENV=mmd_flow before submitting.
if [[ -n ${DMMD_CONDA_ENV:-} ]]; then
  CONDA_HOOK=${DMMD_CONDA_HOOK:-$HOME/miniconda3/condabin/conda}
  eval "$("$CONDA_HOOK" shell.bash hook)"
  conda activate "$DMMD_CONDA_ENV"
fi

PYTHON_BIN=${PYTHON_BIN:-python}
NGPU=${NGPU:-${NSLOTS:-1}}
if [[ "$NGPU" == 4 ]]; then
  CONFIG=${CONFIG:-configs/cifar10_4gpu.yaml}
else
  CONFIG=${CONFIG:-configs/pilot.yaml}
fi
RUN_NAME=${RUN_NAME:-cifar10_pilot_${JOB_ID:-local}}
WORKDIR=${WORKDIR:-runs/$RUN_NAME}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTHONUNBUFFERED=1

args=(--config "$CONFIG" --workdir "$WORKDIR" --set data.num_workers=2)
if [[ -n ${DATA_ROOT:-} ]]; then
  args+=(--data-root "$DATA_ROOT")
fi
if [[ -n ${RESUME:-} ]]; then
  args+=(--resume "$RESUME")
fi
echo "Job: ${JOB_ID:-local}; GPUs: $NGPU; workdir: $WORKDIR"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable in this environment"'

if [[ "$NGPU" == 1 ]]; then
  "$PYTHON_BIN" -u train.py "${args[@]}"
elif [[ "$NGPU" == 2 || "$NGPU" == 4 ]]; then
  # Select a free rendezvous port, avoiding concurrent-job collisions.
  "$PYTHON_BIN" -u -m torch.distributed.run --standalone \
    --nnodes=1 --nproc_per_node="$NGPU" train.py "${args[@]}"
else
  echo "This launcher supports one, two or four GPUs, got $NGPU." >&2
  exit 1
fi
