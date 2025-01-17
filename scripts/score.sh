#!/bin/bash

# Default values
model="resnet50"
dataset="cifar10"

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --model) model="$2"; shift ;;
    --dataset) dataset="$2"; shift ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
  shift
done

savedir="exp/${model}_${dataset}"

command="python3 score.py --savedir $savedir --dataset $dataset"

echo "$command"
eval $command