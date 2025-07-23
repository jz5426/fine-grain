#!/bin/bash

#SBATCH -A mcintoshgroup_gpu
#SBATCH --reservation=mcintoshgroup_gpu1
#SBATCH -t 70:00:00
#SBATCH --mem=40G
#SBATCH -J mgca_exp
#SBATCH -p gpu
#SBATCH -c 10
#SBATCH -N 1
#SBATCH --gres=gpu:l40:1
#SBATCH --begin=now

source activate ctclip

# MGCA

python /cluster/home/t135419uhn/fine-grain/eval.py --model mgca_resnet_50.ckpt  --few_shot 0.01 --learning_rate 0.05 --fusion_type subtraction
python /cluster/home/t135419uhn/fine-grain/eval.py --model mgca_resnet_50.ckpt --few_shot 0.05 --learning_rate 0.05 --fusion_type subtraction
python /cluster/home/t135419uhn/fine-grain/eval.py --model mgca_resnet_50.ckpt --few_shot 0.10 --learning_rate 0.05 --fusion_type subtraction
python /cluster/home/t135419uhn/fine-grain/eval.py --model mgca_resnet_50.ckpt --few_shot 0.20 --learning_rate 0.05 --fusion_type subtraction
python /cluster/home/t135419uhn/fine-grain/eval.py --model mgca_resnet_50.ckpt --few_shot 0.50 --learning_rate 0.05 --fusion_type subtraction
python /cluster/home/t135419uhn/fine-grain/eval.py --model mgca_resnet_50.ckpt --few_shot 0.80 --learning_rate 0.05 --fusion_type subtraction
python /cluster/home/t135419uhn/fine-grain/eval.py --model mgca_resnet_50.ckpt --few_shot 1 --learning_rate 0.05 --fusion_type subtraction

python /cluster/home/t135419uhn/fine-grain/eval.py --model mgca_resnet_50.ckpt --few_shot 0.01 --learning_rate 0.05 --fusion_type concatenate
python /cluster/home/t135419uhn/fine-grain/eval.py --model mgca_resnet_50.ckpt --few_shot 0.05 --learning_rate 0.05 --fusion_type concatenate
python /cluster/home/t135419uhn/fine-grain/eval.py --model mgca_resnet_50.ckpt --few_shot 0.10 --learning_rate 0.05 --fusion_type concatenate
python /cluster/home/t135419uhn/fine-grain/eval.py --model mgca_resnet_50.ckpt --few_shot 0.20 --learning_rate 0.05 --fusion_type concatenate
python /cluster/home/t135419uhn/fine-grain/eval.py --model mgca_resnet_50.ckpt --few_shot 0.50 --learning_rate 0.05 --fusion_type concatenate
python /cluster/home/t135419uhn/fine-grain/eval.py --model mgca_resnet_50.ckpt --few_shot 0.8 --learning_rate 0.05 --fusion_type concatenate
python /cluster/home/t135419uhn/fine-grain/eval.py --model mgca_resnet_50.ckpt --few_shot 1 --learning_rate 0.05 --fusion_type concatenate

python /cluster/home/t135419uhn/fine-grain/eval.py --model mgca_resnet_50.ckpt --few_shot 0.01 --learning_rate 0.05 --fusion_type addition
python /cluster/home/t135419uhn/fine-grain/eval.py --model mgca_resnet_50.ckpt --few_shot 0.05 --learning_rate 0.05 --fusion_type addition
python /cluster/home/t135419uhn/fine-grain/eval.py --model mgca_resnet_50.ckpt --few_shot 0.10 --learning_rate 0.05 --fusion_type addition
python /cluster/home/t135419uhn/fine-grain/eval.py --model mgca_resnet_50.ckpt --few_shot 0.20 --learning_rate 0.05 --fusion_type addition
python /cluster/home/t135419uhn/fine-grain/eval.py --model mgca_resnet_50.ckpt --few_shot 0.50 --learning_rate 0.05 --fusion_type addition
python /cluster/home/t135419uhn/fine-grain/eval.py --model mgca_resnet_50.ckpt --few_shot 0.8 --learning_rate 0.05 --fusion_type addition
python /cluster/home/t135419uhn/fine-grain/eval.py --model mgca_resnet_50.ckpt --few_shot 1 --learning_rate 0.05 --fusion_type addition

# TEXT ONLY
python /cluster/home/t135419uhn/fine-grain/eval.py --model mgca_resnet_50.ckpt  --few_shot 0.01 --learning_rate 0.05 --fusion_type text_only
python /cluster/home/t135419uhn/fine-grain/eval.py --model mgca_resnet_50.ckpt --few_shot 0.05 --learning_rate 0.05 --fusion_type text_only
python /cluster/home/t135419uhn/fine-grain/eval.py --model mgca_resnet_50.ckpt --few_shot 0.10 --learning_rate 0.05 --fusion_type text_only
python /cluster/home/t135419uhn/fine-grain/eval.py --model mgca_resnet_50.ckpt --few_shot 0.20 --learning_rate 0.05 --fusion_type text_only
python /cluster/home/t135419uhn/fine-grain/eval.py --model mgca_resnet_50.ckpt --few_shot 0.50 --learning_rate 0.05 --fusion_type text_only
python /cluster/home/t135419uhn/fine-grain/eval.py --model mgca_resnet_50.ckpt --few_shot 0.80 --learning_rate 0.05 --fusion_type text_only
python /cluster/home/t135419uhn/fine-grain/eval.py --model mgca_resnet_50.ckpt --few_shot 1 --learning_rate 0.05 --fusion_type text_only
