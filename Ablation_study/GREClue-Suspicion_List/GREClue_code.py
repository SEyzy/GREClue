import torch
import argparse
from argparse import ArgumentParser
import os
import pytorch_lightning as pl
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics.cluster import contingency_matrix, fowlkes_mallows_score
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any
import json, re
from pathlib import Path

from src.clustering_models.clusternet_modules.clusternetasmodel import ClusterNetModel
from src.utils import check_args, cluster_acc


def parse_minimal_args(parser):
    parser.add_argument("--dir", default="./pretrained_embeddings/umap_embedded_datasets/", help="dataset directory")
    parser.add_argument("--dataset", default="custom")
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate (default: 1e-4)")
    parser.add_argument("--batch-size", type=int, default=128, help="input batch size for training")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--n-jobs", type=int, default=1, help="number of jobs to run in parallel")
    parser.add_argument("--device", type=str, default="cuda", help="device for computation (default: cpu)")
    parser.add_argument("--offline", action="store_true", help="Run training without Neptune Logger")
    parser.add_argument("--tag", type=str, default="MNIST_UMAPED")
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--limit_train_batches", type=float, default=1., help="used for debugging")
    parser.add_argument("--limit_val_batches", type=float, default=1., help="used for debugging")
    parser.add_argument("--save_checkpoints", type=bool, default=False)
    parser.add_argument("--exp_name", type=str, default="default_exp")
    parser.add_argument(
        "--use_labels_for_eval",
        default=True,
        action="store_true",
        help="whether to use labels for evaluation"
    )
    return parser


def run_on_embeddings_hyperparams(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument("--init_k", default=1, type=int, help="number of initial clusters")
    parser.add_argument("--clusternet_hidden", type=int, default=50)
    parser.add_argument("--clusternet_hidden_layer_list", type=int, nargs="+", default=[50])
    parser.add_argument(
        "--transform_input_data",
        type=str,
        default="normalize",
        choices=["normalize", "min_max", "standard", "standard_normalize", "None", None],
    )
    parser.add_argument("--cluster_loss_weight", type=float, default=1)
    parser.add_argument("--init_cluster_net_weights", action="store_true", default=False)
    parser.add_argument("--when_to_compute_mu", type=str, choices=["once", "every_epoch", "every_5_epochs"], default="every_epoch")
    parser.add_argument("--how_to_compute_mu", type=str, choices=["kmeans", "soft_assign"], default="soft_assign")
    parser.add_argument("--how_to_init_mu", type=str, choices=["kmeans", "soft_assign", "kmeans_1d"], default="kmeans")
    parser.add_argument("--how_to_init_mu_sub", type=str, choices=["kmeans", "soft_assign", "kmeans_1d"], default="kmeans_1d")
    parser.add_argument("--log_emb_every", type=int, default=20)
    parser.add_argument("--log_emb", type=str, default="never", choices=["every_n_epochs", "only_sampled", "never"])
    parser.add_argument("--train_cluster_net", type=int, default=300, help="Number of epochs to pretrain the cluster net")
    parser.add_argument("--cluster_lr", type=float, default=0.0005)
    parser.add_argument("--subcluster_lr", type=float, default=0.005)
    parser.add_argument("--lr_scheduler", type=str, default="StepLR", choices=["StepLR", "None", "ReduceOnP"])
    parser.add_argument("--start_sub_clustering", type=int, default=45)
    parser.add_argument("--subcluster_loss_weight", type=float, default=1.0)
    parser.add_argument("--start_splitting", type=int, default=55)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--softmax_norm", type=float, default=1)
    parser.add_argument("--subcluster_softmax_norm", type=float, default=1)
    parser.add_argument("--split_prob", type=float, default=None)
    parser.add_argument("--merge_prob", type=float, default=None)
    parser.add_argument("--init_new_weights", type=str, default="same", choices=["same", "random", "subclusters"])
    parser.add_argument("--start_merging", type=int, default=55)
    parser.add_argument("--merge_init_weights_sub", type=str, default="highest_ll")
    parser.add_argument("--split_init_weights_sub", type=str, default="random", choices=["same_w_noise", "same", "random"])
    parser.add_argument("--split_every_n_epochs", type=int, default=10)
    parser.add_argument("--split_merge_every_n_epochs", type=int, default=30)
    parser.add_argument("--merge_every_n_epochs", type=int, default=10)
    parser.add_argument("--raise_merge_proposals", type=str, default="brute_force_NN")
    parser.add_argument("--cov_const", type=float, default=0.005)
    parser.add_argument("--freeze_mus_submus_after_splitmerge", type=int, default=5)
    parser.add_argument("--freeze_mus_after_init", type=int, default=5)
    parser.add_argument("--use_priors", type=int, default=1)
    parser.add_argument("--prior", type=str, default="NIW", choices=["NIW", "NIG"])
    parser.add_argument("--pi_prior", type=str, default="uniform", choices=["uniform", None])
    parser.add_argument("--prior_dir_counts", type=float, default=0.1)
    parser.add_argument("--prior_kappa", type=float, default=0.0001)
    parser.add_argument("--NIW_prior_nu", type=float, default=None)
    parser.add_argument("--prior_mu_0", type=str, default="data_mean")
    parser.add_argument("--prior_sigma_choice", type=str, default="isotropic", choices=["iso_005", "iso_001", "iso_0001", "data_std"])
    parser.add_argument("--prior_sigma_scale", type=float, default=".005")
    parser.add_argument("--prior_sigma_scale_step", type=float, default=1.)
    parser.add_argument("--compute_params_every", type=int, default=1)
    parser.add_argument("--start_computing_params", type=int, default=25)
    parser.add_argument("--cluster_loss", type=str, default="KL_GMM_2", choices=["diag_NIG", "isotropic", "KL_GMM_2"])
    parser.add_argument("--subcluster_loss", type=str, default="isotropic", choices=["diag_NIG", "isotropic", "KL_GMM_2"])
    parser.add_argument("--ignore_subclusters", type=bool, default=False)
    parser.add_argument("--log_metrics_at_train", type=bool, default=True)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--evaluate_every_n_epochs", type=int, default=5, help="How often to evaluate the net")
    return parser


def read_seq_list_file(path: str):
    seq = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if "," in s and (";" not in s) and ("#" not in s):
                parts = [p.strip() for p in s.split(",") if p.strip()]
                seq.extend(parts)
            else:
                seq.append(s)
    return seq


def load_failure_groups_seq_only(data_dirs):
    seq_lists, labels, sample_ids = [], [], []
    for root in map(Path, data_dirs):
        if not root.exists():
            print(f"[WARN] missing dir: {root}")
            continue
        for lf in sorted(root.glob("*_list.txt")):
            prefix = lf.name[:-9]
            seq = read_seq_list_file(str(lf))
            seq_lists.append(seq)
            labels.append(-1)
            sample_ids.append(f"{root.name}/{prefix}")
    return seq_lists, labels, sample_ids


class FailureCaseDataset(Dataset):
    def __init__(self, items: List[Dict[str, Any]]):
        self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]


def build_items_from_seq_only(seq_lists: List[List[str]], labels: List[int]):
    items = []
    for seq, y in zip(seq_lists, labels):
        items.append({
            "seq_text": " [SEP] ".join(seq),
            "label": y if y is not None else -1
        })
    return items


def collate_seq_only(batch: List[Dict[str, Any]]):
    seq_texts = [b["seq_text"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return {"seq_texts": seq_texts, "labels": labels}


def _pairwise_prf1_jaccard(y_true: np.ndarray, y_pred: np.ndarray):

    cm = contingency_matrix(y_true, y_pred, sparse=False)

    def comb2(arr):
        arr = arr.astype(np.float64, copy=False)
        return float(((arr * (arr - 1.0)) / 2.0).sum())

    TP = comb2(cm)
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    FP = comb2(col_sums) - TP
    FN = comb2(row_sums) - TP

    pr = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rr = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * pr * rr / (pr + rr)) if (pr + rr) > 0 else 0.0
    jc = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
    return pr, rr, f1, jc


def train_cluster_net():
    parser = argparse.ArgumentParser(description="Seq-only ClusterNet")
    parser = parse_minimal_args(parser)
    parser = run_on_embeddings_hyperparams(parser)

    parser.add_argument("--starcoder_model_name", type=str, default="bigcode/starcoderbase-3b")
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--freeze_starcoder", action="store_true", default=False)

    parser.add_argument("--fusion_hidden_dim", type=int, default=4096)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--data_dirs", nargs="+", required=True)

    args = parser.parse_args()
    args.train_cluster_net = args.max_epochs

    seq_lists, labels, sample_ids = load_failure_groups_seq_only(args.data_dirs)
    items = build_items_from_seq_only(seq_lists, labels)

    split = max(1, int(len(items) * 0.1))
    traindataset = FailureCaseDataset(items[split:])
    valdataset   = FailureCaseDataset(items[:split])

    train_loader = DataLoader(traindataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_seq_only)
    val_loader   = DataLoader(valdataset, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_seq_only)

    check_args(args, args.fusion_hidden_dim)

    if args.seed:
        pl.utilities.seed.seed_everything(args.seed)

    model = ClusterNetModel(
        hparams=args,
        input_dim=args.fusion_hidden_dim,
        init_k=args.init_k
    )

    trainer = pl.Trainer(max_epochs=args.max_epochs, gpus=args.gpus, num_sanity_val_steps=0)
    trainer.fit(model, train_loader, val_loader)

    print("Finished training!")

    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        logits, _codes = model(batch)
        net_pred = logits.argmax(dim=1).cpu().numpy()

        if args.use_labels_for_eval:
            labels_np = batch["labels"].cpu().numpy()
            mask = labels_np > -1
            if mask.sum() >= 2:
                y_true = labels_np[mask]
                y_pred = net_pred[mask]

                fmi = float(fowlkes_mallows_score(y_true, y_pred))
                pr, rr, f1, jc = _pairwise_prf1_jaccard(y_true, y_pred)

                print(f"Val (pairwise): FMI={fmi:.5f} JC={jc:.5f} PR={pr:.5f} RR={rr:.5f} F1={f1:.5f}")
            else:
                print("Val: insufficient labeled samples for pairwise metrics (need at least 2 labeled).")
    return net_pred


if __name__ == "__main__":
    train_cluster_net()
