#!/bin/bash

# Default values
model="resnet50"
dataset="cifar10"
eps=100.0

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --model) model="$2"; shift ;;
    --dataset) dataset="$2"; shift ;;
    --eps) eps="$2"; shift ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
  shift
done

savedir="exp/${model}_${dataset}"

command="python3 plot.py --savedir $savedir --eps $eps"

echo "$command"
eval $command