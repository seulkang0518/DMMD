#!/usr/bin/env bash
#$ -N dmmd_cifar10_4g
#$ -P aihub_ucl
#$ -cwd
#$ -V
#$ -l gpu=true,gpu_type=h100
#$ -pe gpu 4
#$ -l tmem=16G
#$ -l h_rt=24:00:00
#$ -R y
#$ -j y
#$ -o .
#$ -notify

set -eo pipefail
PROJECT_DIR=${DMMD_PROJECT_DIR:-${SGE_O_WORKDIR:-$PWD}}
export NGPU=4
export CONFIG=${CONFIG:-configs/cifar10_4gpu.yaml}
exec bash "$PROJECT_DIR/scripts/train_sge.sh"
