#!/bin/bash

# Default values
model="resnet50"
dataset="cifar10"
gpu_id=0
eps=5.0

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --model) model="$2"; shift ;;
    --dataset) dataset="$2"; shift ;;
    --gpu_id) gpu_id="$2"; shift ;;
    --eps) eps="$2"; shift ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
  shift
done

exp_id="upstream"
log_file="logs/${model}_${dataset}/log_inf_${exp_id}"
savedir="exp/${model}_${dataset}/${exp_id}_eps_${eps}"

echo "Starting inference experiment $exp_id on GPU $gpu_id"

command="CUDA_VISIBLE_DEVICES=$gpu_id python3 -u inference.py \
  --model $model --dataset $dataset \
  --mode upstream --eps $eps\
  --sample_size 1000 --seed 42 \
  --batch_size 32 \
  --savedir=$savedir &> $log_file &"

echo "$command"
eval $command
