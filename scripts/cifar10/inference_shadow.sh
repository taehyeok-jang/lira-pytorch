#!/bin/bash

n_shadows=128
gpus=(0 1 2 3 4 5 6 7)
exp_per_gpu=$((n_shadows / ${#gpus[@]}))

model="resnet50"
dataset="cifar10"

for ((i=0; i<$n_shadows; i++)); do
  gpu_id=${gpus[$((i % ${#gpus[@]}))]}  
  exp_id=$i  
  log_file="logs/${model}_${dataset}/log_inf_${exp_id}"
  savedir="exp/${model}_${dataset}/${exp_id}"

  echo "Starting inference experiment $exp_id on GPU $gpu_id"

  command="CUDA_VISIBLE_DEVICES=$gpu_id python3 -u inference.py \
    --model $model --dataset $dataset \
    --mode shadow \
    --sample_size 1000 --seed 42 \
    --savedir=$savedir &> $log_file &"

  echo "$command"
  eval $command

  if (( (i + 1) % ${#gpus[@]} == 0 )); then
    wait  
  fi
done

