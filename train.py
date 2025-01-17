# PyTorch implementation of
# https://github.com/tensorflow/privacy/blob/master/research/mi_lira_2021/train.py
#
# author: Chenxiang Zhang (orientino)

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import CIFAR10, CIFAR100
from tqdm import tqdm

from wide_resnet import WideResNet

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="resnet50", type=str)
parser.add_argument("--dataset", default="none", type=str)
parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--epochs", default=1, type=int)
parser.add_argument("--n_shadows", default=16, type=int)
parser.add_argument("--shadow_id", default=1, type=int)
parser.add_argument("--pkeep", default=0.5, type=float)
parser.add_argument("--savedir", default="exp/cifar10", type=str)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2023, 0.1994, 0.2010)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)


def run():
    seed = np.random.randint(0, 1000000000)
    seed ^= int(time.time())
    pl.seed_everything(seed)

    args.debug = True
    wandb.init(project="lira", mode="disabled" if args.debug else "online")
    wandb.config.update(args)

    # Dataset
    if args.dataset == "cifar10":
        _mean = cifar10_mean
        _std = cifar10_std
    elif args.dataset == "cifar100":
        _mean = cifar100_mean
        _std = cifar100_std
    else: 
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(_mean, _std),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(_mean, _std),
        ]
    )
    datadir = Path().home() / "dataset"
    print(f"import {args.dataset}...")
    if args.dataset == "cifar10":
        train_ds = CIFAR10(root=datadir, train=True, download=True, transform=train_transform)
        test_ds = CIFAR10(root=datadir, train=False, download=True, transform=test_transform)
    elif args.dataset == "cifar100":
        train_ds = CIFAR100(root=datadir, train=True, download=True, transform=train_transform)
        test_ds = CIFAR100(root=datadir, train=False, download=True, transform=test_transform)
    else:
        raise ValueError("undefined dataset")
    
    # Compute the IN / OUT subset:
    # If we run each experiment independently then even after a lot of trials
    # there will still probably be some examples that were always included
    # or always excluded. So instead, with experiment IDs, we guarantee that
    # after `args.n_shadows` are done, each example is seen exactly half
    # of the time in train, and half of the time not in train.

    size = len(train_ds)
    np.random.seed(seed)
    if args.n_shadows is not None:
        np.random.seed(0)
        keep = np.random.uniform(0, 1, size=(args.n_shadows, size))
        order = keep.argsort(0)
        keep = order < int(args.pkeep * args.n_shadows)
        keep = np.array(keep[args.shadow_id], dtype=bool)
        keep = keep.nonzero()[0]
    else:
        keep = np.random.choice(size, size=int(args.pkeep * size), replace=False)
        keep.sort()
    keep_bool = np.full((size), False)
    keep_bool[keep] = True

    train_ds = torch.utils.data.Subset(train_ds, keep)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4)

    # Model

    print('==> Building model..')
    if args.dataset == "cifar10":
        n_classes = 10 
    elif args.dataset == "cifar100":
        n_classes = 100
    else: 
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    if args.model == 'resnet50':
        m = models.resnet50(pretrained=True)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    else:
        raise NotImplementedError
    
    m = m.to(DEVICE)

    # optim = torch.optim.SGD(m.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)  # ASIS
    optim = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=0.01)                # TOBE
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    # Train
    for i in range(args.epochs):
        m.train()
        loss_total = 0
        pbar = tqdm(train_dl)
        for itr, (x, y) in enumerate(pbar):
            x, y = x.to(DEVICE), y.to(DEVICE)

            loss = F.cross_entropy(m(x), y)
            loss_total += loss

            pbar.set_postfix_str(f"loss: {loss:.2f}")
            optim.zero_grad()
            loss.backward()
            optim.step()
        sched.step()

        wandb.log({"loss": loss_total / len(train_dl)})

        print(f"[test] acc_test: {get_acc(m, test_dl):.4f}")

    wandb.log({"acc_test": get_acc(m, test_dl)})

    savedir = os.path.join(args.savedir, str(args.shadow_id))
    os.makedirs(savedir, exist_ok=True)
    np.save(savedir + "/keep.npy", keep_bool)
    torch.save(m.state_dict(), savedir + "/model.pt")


@torch.no_grad()
def get_acc(model, dl):
    acc = []
    for x, y in dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        acc.append(torch.argmax(model(x), dim=1) == y)
    acc = torch.cat(acc)
    acc = torch.sum(acc) / len(acc)

    return acc.item()


if __name__ == "__main__":
    run()
