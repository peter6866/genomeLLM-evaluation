#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH -o log/llm_%j.out
#SBATCH -e log/llm_%j.err
#SBATCH --mail-type=END,FAIL

python3 fit_llm.py