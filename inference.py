# PyTorch implementation of
# https://github.com/tensorflow/privacy/blob/master/research/mi_lira_2021/inference.py
#
# author: Chenxiang Zhang (orientino)

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from torchvision.datasets import CIFAR10, CIFAR100
from tqdm import tqdm

from io import BytesIO
from PIL import Image
import requests

from wide_resnet import WideResNet

parser = argparse.ArgumentParser()
parser.add_argument("--n_queries", default=1, type=int)
parser.add_argument("--model", default="resnet50", type=str)
parser.add_argument("--dataset", default="none", type=str)
parser.add_argument("--mode", default="none", type=str, help="shadow or upstream")
parser.add_argument("--eps", default=-1.0, type=float, help="epsilon; the level of defense from upstream")
parser.add_argument("--sample_size", default=1000, type=int)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--savedir", default="exp/cifar10", type=str)
args = parser.parse_args()

if args.mode == "none": 
    raise ValueError(f"mode must be specified")
if args.mode == "upstream" and args.eps == -1.0:
    raise ValueError(f"if mode = upstream , eps (epsilon) must be specified: {args.eps}")


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2023, 0.1994, 0.2010)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")

@torch.no_grad()
def run():

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    global _accuracy, _latency # model spec 
    global _mean, _std

    # For resnet56 (as victim model in model zoo)
    if args.dataset == "cifar10":
        _accuracy, _latency = 94.3, 42.77
    elif args.dataset == "cifar100":
        _accuracy, _latency = 75.16, 41.82
    else: 
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Dataset
    if args.dataset == "cifar10":
        _mean = cifar10_mean
        _std = cifar10_std
    elif args.dataset == "cifar100":
        _mean = cifar100_mean
        _std = cifar100_std
    else: 
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(_mean, _std),
        ]
    )
    
    print(f"import {args.dataset}...")
    datadir = Path().home() / "dataset"

    if args.dataset == "cifar10":
        full_ds = CIFAR10(root=datadir, train=True, download=True, transform=transform)
    elif args.dataset == "cifar100":
        full_ds = CIFAR100(root=datadir, train=True, download=True, transform=transform)
    else:
        raise ValueError("Undefined dataset")
    
    indices = np.random.choice(len(full_ds), args.sample_size, replace=False)
    print(indices[:100])

    sampled_ds = Subset(full_ds, indices)
    sample_dl = DataLoader(sampled_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    if args.mode == "shadow":
        inference_w_shadow(sample_dl=sample_dl)
    elif args.mode == "upstream":
        inference_upstream(sample_dl=sample_dl)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


def inference_w_shadow(sample_dl): 

    # Infer the logits with multiple queries
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
    m.load_state_dict(torch.load(os.path.join(args.savedir, "model.pt")))
    m.to(DEVICE)
    m.eval()

    logits_n = []
    for i in range(args.n_queries):
        logits = []
        for x, _ in tqdm(sample_dl):
            x = x.to(DEVICE)
            outputs = m(x)
            logits.append(outputs.cpu().numpy())
        logits_n.append(np.concatenate(logits))
    logits_n = np.stack(logits_n, axis=1)
    print(logits_n.shape)

    print(f"Saved logits.npy and keep.npy to {args.savedir}")

    np.save(os.path.join(args.savedir, "logits.npy"), logits_n)

    sampled_indices = sample_dl.dataset.indices  
    keep = np.load(os.path.join(args.savedir, "keep.npy"))  # Load full keep array
    keep_sampled = keep[sampled_indices]

    np.save(os.path.join(args.savedir, "keep_sampled.npy"), keep_sampled)


def inference_upstream(sample_dl): 

    logits_n = []
    for i in range(args.n_queries):
        logits = []
        for x, _ in tqdm(sample_dl):
            x = x.to(DEVICE)
            """""""""""""""
            request to upstream. 
            """""""""""""""
            # outputs = m(x)
            image_payloads = []
            for img in x:
                img = img.cpu().numpy().transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
                img = (img * np.array(_std) + np.array(_mean))  # reverse-normalization
                img = (img * 255).clip(0, 255).astype(np.uint8)  # convert to 0-255
                buffer = BytesIO()
                Image.fromarray(img).save(buffer, format="JPEG")
                buffer.seek(0)
                image_payloads.append(("files", ("image.jpg", buffer, "image/jpeg")))

            try:
                response = requests.post(
                    f"http://127.0.0.1:8000/mz/b_secure_classify_?accuracy={_accuracy}&latency={_latency}&eps={args.eps}",
                    files=image_payloads
                )
                response.raise_for_status()
                batch_outputs = response.json()  

                print(batch_outputs)

                logits.extend([output["outputs"] for output in batch_outputs])

            except requests.exceptions.RequestException as e:
                print(f"API call failed: {e}")
                logits.extend([[0] * 10] * len(x))  # dummy 
            finally:
                for _, (_, buffer, _) in image_payloads:
                    buffer.close()

        logits_n.append(np.array(logits))

    logits_n = np.stack(logits_n, axis=1)
    print(logits_n.shape)

    # Save logits
    print(f"Saved logits and keep.npy to {args.savedir}")

    os.makedirs(args.savedir, exist_ok=True)
    
    np.save(os.path.join(args.savedir, "logits.npy"), logits_n)

    keep = np.ones(len(sample_dl.dataset), dtype=bool)
    np.save(os.path.join(args.savedir, "keep_sampled.npy"), keep)
    

if __name__ == "__main__":
    run()
