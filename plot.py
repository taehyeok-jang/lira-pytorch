# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified copy by Chenxiang Zhang (orientino) of the original:
# https://github.com/tensorflow/privacy/tree/master/research/mi_lira_2021


import argparse
import functools
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy.stats
from sklearn.metrics import auc, roc_curve

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

parser = argparse.ArgumentParser()
parser.add_argument("--sample_size", default=1000, type=int)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--eps", default=-1.0, type=float, help="epsilon; the level of defense from upstream")
parser.add_argument("--savedir", default="exp/cifar10", type=str)
args = parser.parse_args()

if args.eps == -1.0:
    raise ValueError(f"eps (epsilon) must be specified: {args.eps}")


def sweep(score, x):
    """
    Compute a ROC curve and then return the FPR, TPR, AUC, and ACC.
    """
    fpr, tpr, _ = roc_curve(x, -score)
    acc = np.max(1 - (fpr + (1 - tpr)) / 2)
    return fpr, tpr, auc(fpr, tpr), acc


def load_data():
    """
    Load our saved scores and then put them into a big matrix.
    """
    global scores, keep, upstream_scores, upstream_keep
    scores = []
    keep = []

    print("load_data...")

    for path in os.listdir(args.savedir):
        if path.startswith("upstream"):
            continue
        scores.append(np.load(os.path.join(args.savedir, path, "scores.npy")))
        keep.append(np.load(os.path.join(args.savedir, path, "keep_sampled.npy")))
    scores = np.array(scores)
    keep = np.array(keep)

    upstream_scores = np.load(os.path.join(args.savedir, f"upstream_eps_{args.eps}", "scores.npy"))
    upstream_keep = np.load(os.path.join(args.savedir, f"upstream_eps_{args.eps}", "keep_sampled.npy"))

    return scores, keep, upstream_scores, upstream_keep


def generate_ours(keep, scores, check_keep, check_scores, in_size=100000, out_size=100000, fix_variance=False):
    """
    Fit a two predictive models using keep and scores in order to predict
    if the examples in check_scores were training data or not, using the
    ground truth answer from check_keep.

    keep: (128, 1000)
    scores:  (128, 1000, 1)
    check_keep:  (1, 1000)
    check_scores:  (1, 1000, 1)
    len(dat_in):  1000
    len(dat_out):  1000
    dat_in[0]:  (64, 1)
    dat_out[0]:  (64, 1)
    dat_in:  (1000, 64, 1)
    dat_out:  (1000, 63, 1)

    """
    print("generate_ours...")

    check_keep = check_keep.reshape(1, -1)
    check_scores = check_scores.reshape(1, check_scores.shape[0], -1)

    dat_in = []
    dat_out = []

    for j in range(scores.shape[1]):
        dat_in.append(scores[keep[:, j], j, :])
        dat_out.append(scores[~keep[:, j], j, :])

    in_size = min(min(map(len, dat_in)), in_size)
    out_size = min(min(map(len, dat_out)), out_size)

    dat_in = np.array([x[:in_size] for x in dat_in])
    dat_out = np.array([x[:out_size] for x in dat_out])

    mean_in = np.median(dat_in, 1)
    mean_out = np.median(dat_out, 1)

    if fix_variance:
        std_in = np.std(dat_in)
        std_out = np.std(dat_in)
    else:
        std_in = np.std(dat_in, 1)
        std_out = np.std(dat_out, 1)

    prediction = []
    answers = []
    for ans, sc in zip(check_keep, check_scores):
        pr_in = -scipy.stats.norm.logpdf(sc, mean_in, std_in + 1e-30)
        pr_out = -scipy.stats.norm.logpdf(sc, mean_out, std_out + 1e-30)
        score = pr_in - pr_out

        prediction.extend(score.mean(1))
        answers.extend(ans)

    print('prediction: ', prediction[:10])
    print('answers: ', answers[:10])
    
    l_thresholds = np.arange(-2, 2.5, 0.5)
    print(f'l_threshold - accuracy:')
    for th in l_thresholds:
        # Calculate accuracy on various l_threshold
        predicted_labels = [p >= th for p in prediction]
        accuracy = np.mean([p == a for p, a in zip(predicted_labels, answers)])
        print(f'{th}\t{accuracy}')

    # Reshape check_scores to match the expected dimensions
    check_scores = check_scores.reshape(-1)

    x = np.linspace(-20, 20, 500)  # Fix the range of x to [-20, 20]
    pdf_in = scipy.stats.norm.pdf(x, np.mean(mean_in), np.mean(std_in))
    pdf_out = scipy.stats.norm.pdf(x, np.mean(mean_out), np.mean(std_out))
    print(f'~N(mean_in={np.mean(mean_in):.2f}, std_in={np.mean(std_in):.2f})')
    print(f'~N(mean_out={np.mean(mean_out):.2f}, std_out={np.mean(std_out):.2f})')

    # Plot the histogram and normal distributions
    plt.figure(figsize=(10, 6))
    plt.hist(check_scores, bins=30, alpha=0.6, edgecolor='black', density=True, label=f'check_scores (eps={args.eps})')
    plt.plot(x, pdf_in, label=f'Q_in (member)', lw=2)
    plt.plot(x, pdf_out, label=f'Q_out (non-member)', lw=2, linestyle='--')

    plt.title("Histogram of Check Scores with Normal Distributions")
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"./figures/check_scores_eps_{args.eps}.png", dpi=300, bbox_inches='tight')
    plt.show()

    return prediction, answers


def generate_ours_offline(keep, scores, check_keep, check_scores, in_size=100000, out_size=100000, fix_variance=False):
    """
    Fit a single predictive model using keep and scores in order to predict
    if the examples in check_scores were training data or not, using the
    ground truth answer from check_keep.

    keep: (127, 1000)
    scores:  (127, 1000, 1)
    check_keep:  (1, 1000)
    check_scores:  (1, 1000, 1)
    """

    print("generate_ours_offline...")

    check_keep = check_keep.reshape(1, -1)  # (1000,) -> (1, 1000)
    check_scores = check_scores.reshape(1, check_scores.shape[0], -1)  # (1000, 1) -> (1, 1000, 1)

    dat_in = []
    dat_out = []

    for j in range(scores.shape[1]):
        dat_in.append(scores[keep[:, j], j, :])
        dat_out.append(scores[~keep[:, j], j, :])

    out_size = min(min(map(len, dat_out)), out_size)

    dat_out = np.array([x[:out_size] for x in dat_out])

    mean_out = np.median(dat_out, 1)

    if fix_variance:
        std_out = np.std(dat_out)
    else:
        std_out = np.std(dat_out, 1)

    prediction = []
    answers = []
    for ans, sc in zip(check_keep, check_scores):
        likelihood_ratio = scipy.stats.norm.cdf(sc, mean_out, std_out + 1e-30)
        
        # Aggregate predictions (mean or another aggregation can be applied)
        prediction.extend(likelihood_ratio.mean(1))
        answers.extend(ans)

    print('prediction: ', prediction[:10])
    print('answers: ', answers[:10])
    
    l_thresholds = np.arange(0.0, 1.1, 0.1)
    print(f'l_threshold - accuracy:')
    for th in l_thresholds:
        # Calculate accuracy on various l_threshold
        predicted_labels = [p >= th for p in prediction]
        accuracy = np.mean([p == a for p, a in zip(predicted_labels, answers)])
        print(f'{th}\t{accuracy}')

    # x-axis range for normal distributions
    x = np.linspace(-20, 20, 500)  # Fix the range of x to [-20, 20]

    pdf_out = scipy.stats.norm.pdf(x, np.mean(mean_out), np.mean(std_out))
    cdf_out = scipy.stats.norm.cdf(x, np.mean(mean_out), np.mean(std_out))
    likelihood_ratio = 1 - cdf_out

    # Plot the distributions
    plt.figure(figsize=(10, 6))
    plt.hist(prediction, bins=30, alpha=0.7, edgecolor='black', density=True, label='Predictions', color='orange')

    plt.title("Predictions")
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"./figures/check_scores_offline_eps_{args.eps}.png", dpi=300, bbox_inches='tight')
    plt.show()

    return prediction, answers


def generate_global(keep, scores, check_keep, check_scores):
    """
    Use a simple global threshold sweep to predict if the examples in
    check_scores were training data or not, using the ground truth answer from
    check_keep.
    """
    prediction = []
    answers = []
    for ans, sc in zip(check_keep, check_scores):
        prediction.extend(-sc.mean(1))
        answers.extend(ans)

    return prediction, answers


def do_plot(fn, keep, scores, upstream_keep, upstream_scores, ntest, legend="", metric="auc", sweep_fn=sweep, **plot_kwargs):
    """
    Generate the ROC curves by using ntest models as test models and the rest to train.
    """

    prediction, answers = fn(keep, scores, upstream_keep, upstream_scores)

    fpr, tpr, auc, acc = sweep_fn(np.array(prediction), np.array(answers, dtype=bool))

    low = -1 # tpr[np.where(fpr < 0.001)[0][-1]]

    print("Attack %s   AUC %.4f, Accuracy %.4f, TPR@0.1%%FPR of %.4f" % (legend, auc, acc, low))

    metric_text = ""
    if metric == "auc":
        metric_text = "auc=%.3f" % auc
    elif metric == "acc":
        metric_text = "acc=%.3f" % acc

    plt.plot(fpr, tpr, label=legend + metric_text, **plot_kwargs)
    return (acc, auc)


def fig_fpr_tpr():
    plt.figure(figsize=(4, 3))

    do_plot(generate_ours, keep, scores, upstream_keep, upstream_scores, 1, "Ours (online)\n", metric="auc")

    # do_plot(functools.partial(generate_ours, fix_variance=True), keep, scores, upstream_keep, upstream_scores, 1, "Ours (online, fixed variance)\n", metric="auc")

    do_plot(functools.partial(generate_ours_offline), keep, scores, upstream_keep, upstream_scores, 1, "Ours (offline)\n", metric="auc")

    # do_plot(functools.partial(generate_ours_offline, fix_variance=True), keep, scores, upstream_keep, upstream_scores, 1, "Ours (offline, fixed variance)\n", metric="auc")

    # do_plot(generate_global, keep, scores, upstream_keep, upstream_scores, 1, "Global threshold\n", metric="auc")

    plt.semilogx()
    plt.semilogy()
    plt.xlim(1e-5, 1)
    plt.ylim(1e-5, 1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot([0, 1], [0, 1], ls="--", color="gray")
    plt.subplots_adjust(bottom=0.18, left=0.18, top=0.96, right=0.96)
    plt.legend(fontsize=8)
    plt.savefig("fprtpr.png")
    plt.show()


if __name__ == "__main__":

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    load_data()
    fig_fpr_tpr()
