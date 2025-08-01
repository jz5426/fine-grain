#!/bin/bash

#SBATCH -A mcintoshgroup_gpu
#SBATCH --reservation=mcintoshgroup_gpu1
#SBATCH -t 70:00:00
#SBATCH --mem=40G
#SBATCH -J cxrclip_features
#SBATCH -p gpu
#SBATCH -c 10
#SBATCH -N 1
#SBATCH --gres=gpu:l40:1
#SBATCH --begin=now

source activate ctclip

# rexerr
python /cluster/home/t135419uhn/fine-grain/experiment_scripts/encode_feats_main.py --model r50_m.tar --batch_size 256 --encode_data_only true --eval_dataset rexerr

