#!/bin/bash

#SBATCH -A mcintoshgroup_gpu
#SBATCH --reservation=mcintoshgroup_gpu1
#SBATCH -t 70:00:00
#SBATCH --mem=40G
#SBATCH -J fine_tune_mgca
#SBATCH -p gpu
#SBATCH -c 10
#SBATCH -N 1
#SBATCH --gres=gpu:l40:1
#SBATCH --begin=now

source activate ctclip

python /cluster/home/t135419uhn/fine-grain/experiment_scripts/evaluate_fine_tune_main.py --model r50_m.tar --batch_size 1024 --mask_uncertain_labels true --fine_tune_modal image --max_text_len 256
python /cluster/home/t135419uhn/fine-grain/experiment_scripts/evaluate_fine_tune_main.py --model r50_mcc.tar --batch_size 1024 --mask_uncertain_labels true --fine_tune_modal image --max_text_len 256

python /cluster/home/t135419uhn/fine-grain/experiment_scripts/evaluate_fine_tune_main.py --model r50_m.tar --batch_size 1024 --mask_uncertain_labels true --fine_tune_modal text --max_text_len 256
python /cluster/home/t135419uhn/fine-grain/experiment_scripts/evaluate_fine_tune_main.py --model r50_mcc.tar --batch_size 1024 --mask_uncertain_labels true --fine_tune_modal text --max_text_len 256

python /cluster/home/t135419uhn/fine-grain/experiment_scripts/evaluate_fine_tune_main.py --model mgca_resnet_50.ckpt --batch_size 1024 --mask_uncertain_labels true --fine_tune_modal image --max_text_len 256
python /cluster/home/t135419uhn/fine-grain/experiment_scripts/evaluate_fine_tune_main.py --model mgca_resnet_50.ckpt --batch_size 1024 --mask_uncertain_labels true --fine_tune_modal text --max_text_len 256