#!/bin/bash

# Default values
n_shadows=128
model="resnet50"
dataset="cifar10"
epochs=20
lr=0.001

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --n_shadows) n_shadows="$2"; shift ;;
    --model) model="$2"; shift ;;
    --dataset) dataset="$2"; shift ;;
    --epochs) epochs="$2"; shift ;;
    --lr) lr="$2"; shift ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
  shift
done

gpus=(0 1 2 3 4 5 6 7)
shadows_per_gpu=$((n_shadows / ${#gpus[@]}))
savedir="exp/${model}_${dataset}"

for ((i=0; i<$n_shadows; i++)); do
  gpu_id=${gpus[$((i % ${#gpus[@]}))]} 
  shadow_id=$i 
  log_file="logs/${model}_${dataset}/log_${shadow_id}" 

  echo "Starting shadow experiment $shadow_id on GPU $gpu_id"

  command="CUDA_VISIBLE_DEVICES=$gpu_id python3 -u train.py \
    --model $model --dataset $dataset \
    --epochs=$epochs --lr=$lr \
    --n_shadows=$n_shadows --shadow_id=$shadow_id --debug \
    --savedir=$savedir \
    &> $log_file &"
  
  echo "$command"
  eval $command

  if (( (i + 1) % ${#gpus[@]} == 0 )); then
    wait
  fi
done

wait

