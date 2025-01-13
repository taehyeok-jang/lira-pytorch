model="resnet50"
dataset="cifar100"

savedir="exp/${model}_${dataset}" # just specify ${model}_${dataset} 

command="python3 score.py --savedir $savedir --dataset $dataset"

echo "$command"
eval $command