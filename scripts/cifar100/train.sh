#!/bin/bash

n_shadows=128
gpus=(0 1 2 3 4 5 6 7)
shadows_per_gpu=$((n_shadows / ${#gpus[@]}))

model="resnet50"
dataset="cifar100"
savedir="exp/${model}_${dataset}"

epochs=40
lr=0.001

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

