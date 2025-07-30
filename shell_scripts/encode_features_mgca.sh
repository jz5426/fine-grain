#!/bin/bash

#SBATCH -A mcintoshgroup_gpu
#SBATCH --reservation=mcintoshgroup_gpu1
#SBATCH -t 70:00:00
#SBATCH --mem=40G
#SBATCH -J mgca_features
#SBATCH -p gpu
#SBATCH -c 10
#SBATCH -N 1
#SBATCH --gres=gpu:l40:1
#SBATCH --begin=now

source activate ctclip

# rexerr
# python /cluster/home/t135419uhn/fine-grain/eval.py --model mgca_resnet_50.ckpt --batch_size 256 --text_max_length 512

# mimic cxr
python /cluster/home/t135419uhn/fine-grain/experiment_scripts/evaluate_fine_tune_main.py --model mgca_resnet_50.ckpt --batch_size 256 --encode_data_only true

