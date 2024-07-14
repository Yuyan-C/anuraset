#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --partition=long              
#SBATCH --cpus-per-task=8     
#SBATCH --gres=gpu:rtx8000:1              

module load python/3.9

source $HOME/mothenv/bin/activate

CURDIR=/home/mila/y/yuyan.chen/projects/anuraset

python $CURDIR/baseline/train.py "$@" 




