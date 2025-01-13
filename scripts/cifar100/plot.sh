model="resnet50"
dataset="cifar100"

eps=100.0
savedir="exp/${model}_${dataset}"

command="python3 plot.py --savedir $savedir --eps $eps"

echo "$command"
eval $command