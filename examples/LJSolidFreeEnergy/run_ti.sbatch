#!/bin/bash
#SBATCH --job-name=lj-ti
#SBATCH --output=lj_ti_%j.out
#SBATCH --error=lj_ti_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

module purge
module load gcc/8.2.0
module load ompi/3.1.6/gnu-8.2.0
module load cuda/11.8.0-gcc-7.2.0-xqzqlf2

export MAMBA_ROOT_PREFIX="$HOME/micromamba"
eval "$($MAMBA_ROOT_PREFIX/micromamba shell hook --shell bash)"
micromamba activate mintenv

# reload modules (activating conda breaks them)
module purge
module load gcc/8.2.0
module load ompi/3.1.6/gnu-8.2.0
module load cuda/11.8.0-gcc-7.2.0-xqzqlf2

srun -n 1 python einstein_nvt.py
