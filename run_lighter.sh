python3 train.py --epochs 5 --n_shadow 4 --shadow_id 0 --debug
python3 train.py --epochs 5 --n_shadow 4 --shadow_id 1 --debug
python3 train.py --epochs 5 --n_shadow 4 --shadow_id 2 --debug
python3 train.py --epochs 5 --n_shadow 4 --shadow_id 3 --debug

python3 inference.py --savedir exp/cifar10
python3 score.py --savedir exp/cifar10
python3 plot.py --savedir exp/cifar10

