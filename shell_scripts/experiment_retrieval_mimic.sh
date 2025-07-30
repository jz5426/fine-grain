#!/bin/bash

#SBATCH -A mcintoshgroup_gpu
#SBATCH --reservation=mcintoshgroup_gpu1
#SBATCH -t 70:00:00
#SBATCH --mem=40G
#SBATCH -J retrieval
#SBATCH -p gpu
#SBATCH -c 10
#SBATCH -N 1
#SBATCH --gres=gpu:l40:1
#SBATCH --begin=now

source activate ctclip

python /cluster/home/t135419uhn/fine-grain/experiment_scripts/evaluate_retrieval_main.py --model r50_m.tar --max_text_len 256
python /cluster/home/t135419uhn/fine-grain/experiment_scripts/evaluate_retrieval_main.py --model mgca_resnet_50.ckpt --max_text_len 256

python /cluster/home/t135419uhn/fine-grain/experiment_scripts/evaluate_retrieval_main.py --model r50_m.tar --max_text_len 128
python /cluster/home/t135419uhn/fine-grain/experiment_scripts/evaluate_retrieval_main.py --model mgca_resnet_50.ckpt --max_text_len 128