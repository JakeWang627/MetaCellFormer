#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from datetime import datetime
import uuid
from collections import defaultdict

import numpy as np
import scanpy as sc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


# ================================
# Utils
# ================================
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def to_dense_float32(X):
    X = X.A if hasattr(X, "A") else X
    return np.asarray(X, dtype=np.float32)


# ================================
# Graph builders
# ================================
def build_intra_species_knn(X, species_labels, k=15):
    N = X.shape[0]
    neighbors_intra = [[] for _ in range(N)]
    sp2idx = defaultdict(list)
    for i, sp in enumerate(species_labels):
        sp2idx[int(sp)].append(i)

    for sp, idxs in sp2idx.items():
        if len(idxs) <= 1:
            continue
        Xs = X[idxs]
        kk = min(k + 1, len(idxs))
        nnm = NearestNeighbors(n_neighbors=kk, metric="euclidean", n_jobs=-1).fit(Xs)
        _, nbrs = nnm.kneighbors(Xs, return_distance=True)
        for local_i, neigh in enumerate(nbrs):
            gi = idxs[local_i]
            neighbors_intra[gi] = [idxs[j] for j in neigh[1:]]  # drop self
    return neighbors_intra


def build_cross_species_mnn_by_label(X, labels, species_labels, k=5):
    """Your original: MNN within each cell-type label (stratified by labels)."""
    N = X.shape[0]
    neighbors_cross = [[] for _ in range(N)]
    unique_labels = np.unique(labels)
    unique_species = np.unique(species_labels)

    for lab in unique_labels:
        idx_lab = np.where(labels == lab)[0]
        if len(idx_lab) < 2:
            continue

        sp2idx = {}
        for sp in unique_species:
            idx = idx_lab[species_labels[idx_lab] == sp]
            if len(idx) > 0:
                sp2idx[int(sp)] = idx

        sps = list(sp2idx.keys())
        if len(sps) < 2:
            continue

        for i in range(len(sps)):
            for j in range(i + 1, len(sps)):
                sa, sb = sps[i], sps[j]
                A = sp2idx[sa]
                B = sp2idx[sb]
                if len(A) == 0 or len(B) == 0:
                    continue

                kA = min(k, len(B))
                kB = min(k, len(A))

                nn_AB = NearestNeighbors(n_neighbors=kA, metric="euclidean", n_jobs=-1).fit(X[B])
                idx_AB = nn_AB.kneighbors(X[A], return_distance=False)

                nn_BA = NearestNeighbors(n_neighbors=kB, metric="euclidean", n_jobs=-1).fit(X[A])
                idx_BA = nn_BA.kneighbors(X[B], return_distance=False)

                A_to_B = [set(B[idx_AB[t]]) for t in range(len(A))]
                B_to_A = [set(A[idx_BA[t]]) for t in range(len(B))]
                B_to_A_map = {B[t]: B_to_A[t] for t in range(len(B))}

                for t, a in enumerate(A):
                    for b in A_to_B[t]:
                        if a in B_to_A_map.get(int(b), set()):
                            neighbors_cross[int(a)].append(int(b))
                            neighbors_cross[int(b)].append(int(a))

    neighbors_cross = [sorted(list(set(nbs))) for nbs in neighbors_cross]
    return neighbors_cross


def build_cross_species_mnn_global(X, species_labels, k=5):
    """Ablation #10: global MNN across species (no label stratification)."""
    N = X.shape[0]
    neighbors_cross = [[] for _ in range(N)]
    unique_species = np.unique(species_labels)
    if len(unique_species) < 2:
        return neighbors_cross

    for i in range(len(unique_species)):
        for j in range(i + 1, len(unique_species)):
            sa, sb = unique_species[i], unique_species[j]
            A = np.where(species_labels == sa)[0]
            B = np.where(species_labels == sb)[0]
            if len(A) == 0 or len(B) == 0:
                continue

            kA = min(k, len(B))
            kB = min(k, len(A))

            nn_AB = NearestNeighbors(n_neighbors=kA, metric="euclidean", n_jobs=-1).fit(X[B])
            idx_AB = nn_AB.kneighbors(X[A], return_distance=False)

            nn_BA = NearestNeighbors(n_neighbors=kB, metric="euclidean", n_jobs=-1).fit(X[A])
            idx_BA = nn_BA.kneighbors(X[B], return_distance=False)

            A_to_B = [set(B[idx_AB[t]]) for t in range(len(A))]
            B_to_A = [set(A[idx_BA[t]]) for t in range(len(B))]
            B_to_A_map = {B[t]: B_to_A[t] for t in range(len(B))}

            for t, a in enumerate(A):
                for b in A_to_B[t]:
                    if a in B_to_A_map.get(int(b), set()):
                        neighbors_cross[int(a)].append(int(b))
                        neighbors_cross[int(b)].append(int(a))

    neighbors_cross = [sorted(list(set(nbs))) for nbs in neighbors_cross]
    return neighbors_cross


def randomize_cross_edges(neighbors_cross, species_labels, seed=2025):
    """Ablation #11: keep per-node cross degree, but randomize targets across species."""
    rng = np.random.default_rng(seed)
    N = len(neighbors_cross)

    sp2idx = defaultdict(list)
    for idx, sp in enumerate(species_labels):
        sp2idx[int(sp)].append(idx)
    unique_species = list(sp2idx.keys())

    neighbors_rand = [[] for _ in range(N)]

    for i in range(N):
        sp_i = int(species_labels[i])
        deg_i = len(neighbors_cross[i])
        if deg_i == 0:
            continue

        pool = []
        for sp in unique_species:
            if sp != sp_i:
                pool.extend(sp2idx[sp])
        if len(pool) == 0:
            continue

        if deg_i <= len(pool):
            sampled = rng.choice(pool, size=deg_i, replace=False)
        else:
            sampled = rng.choice(pool, size=deg_i, replace=True)

        neighbors_rand[i] = sampled.astype(int).tolist()

    # symmetric
    for i in range(N):
        for j in neighbors_rand[i]:
            neighbors_rand[j].append(i)

    neighbors_rand = [sorted(list(set(nbs))) for nbs in neighbors_rand]
    return neighbors_rand


# ================================
# Subgraph dataloader
# ================================
class IdxDataset(Dataset):
    def __init__(self, n):
        self.n = n
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        return idx


def make_subgraph_collate(features, labels, species_labels,
                          neighbors_intra, neighbors_cross,
                          n_expand_intra=15, n_expand_cross=5):
    features = np.asarray(features)
    labels = np.asarray(labels)
    species_labels = np.asarray(species_labels)

    def collate(batch_idx_list):
        seed_idx = np.array(batch_idx_list, dtype=np.int64)

        sub_set = set(seed_idx.tolist())
        for g in seed_idx:
            for nb in neighbors_intra[g][:n_expand_intra]:
                sub_set.add(nb)
            for nb in neighbors_cross[g][:n_expand_cross]:
                sub_set.add(nb)

        sub_nodes = np.array(sorted(list(sub_set)), dtype=np.int64)
        pos = {g: i for i, g in enumerate(sub_nodes)}
        seed_pos = np.array([pos[g] for g in seed_idx], dtype=np.int64)

        rows, cols, etype = [], [], []
        for g in sub_nodes:
            src = pos[g]
            for nb in neighbors_intra[g][:n_expand_intra]:
                if nb in pos:
                    rows.append(src); cols.append(pos[nb]); etype.append(0)
            for nb in neighbors_cross[g][:n_expand_cross]:
                if nb in pos:
                    rows.append(src); cols.append(pos[nb]); etype.append(1)

        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        edge_type  = torch.tensor(etype, dtype=torch.long)

        x_sub = torch.tensor(features[sub_nodes], dtype=torch.float32)
        y_seed = torch.tensor(labels[seed_idx], dtype=torch.long)
        sp_seed = torch.tensor(species_labels[seed_idx], dtype=torch.long)

        return x_sub, y_seed, sp_seed, edge_index, edge_type, torch.tensor(seed_pos, dtype=torch.long)

    return collate


# ================================
# Losses
# ================================
class SoftTripleLoss(nn.Module):
    def __init__(self, embedding_dim, n_classes, n_centers=5, la=10.0, gamma=0.1, tau=0.2):
        super().__init__()
        self.la = la
        self.gamma = gamma
        self.tau = tau
        self.n_classes = n_classes
        self.n_centers = n_centers
        self.centers = nn.Parameter(torch.randn(n_classes * n_centers, embedding_dim) / np.sqrt(embedding_dim))
        self.weight = nn.Parameter(torch.ones(n_classes, n_centers) / n_centers)

    def forward(self, x, labels):
        x = F.normalize(x, dim=1)
        centers = F.normalize(self.centers, dim=1)
        sim = torch.matmul(x, centers.t())
        B = x.size(0)
        sim = sim.view(B, self.n_classes, self.n_centers)
        weight = F.softmax(self.weight, dim=1)
        sim_weighted = torch.sum(sim * weight, dim=2)
        mask = torch.zeros_like(sim_weighted).scatter_(1, labels.unsqueeze(1), 1)
        sim_target = torch.sum(sim_weighted * mask, dim=1)
        sim_others = torch.max(sim_weighted * (1 - mask), dim=1)[0]
        loss = torch.mean(F.softplus(self.la * (sim_others - sim_target + self.gamma)))
        return loss


def variance_loss(embeddings, labels):
    unique_labels = labels.unique()
    loss = 0.0
    count = 0
    for lab in unique_labels:
        mask = (labels == lab)
        cluster = embeddings[mask]
        if cluster.size(0) < 2:
            continue
        center = cluster.mean(dim=0, keepdim=True)
        loss += ((cluster - center) ** 2).sum(dim=1).mean()
        count += 1
    if count == 0:
        return torch.tensor(0.0, device=embeddings.device)
    return loss / count


def graph_smoothness_loss(z, edge_index, edge_type, alpha=None):
    src = edge_index[0]; dst = edge_index[1]
    mask = (edge_type == 0)  # intra only
    if mask.sum() == 0:
        return torch.tensor(0.0, device=z.device)
    src = src[mask]; dst = dst[mask]
    diff2 = (z[src] - z[dst]).pow(2).sum(dim=1)
    if alpha is None:
        return diff2.mean()
    a = alpha[mask]
    return (a * diff2).mean()


def mnn_pull_loss(z, edge_index, edge_type, alpha=None):
    src = edge_index[0]; dst = edge_index[1]
    mask = (edge_type == 1)  # cross only
    if mask.sum() == 0:
        return torch.tensor(0.0, device=z.device)
    src = src[mask]; dst = dst[mask]
    diff2 = (z[src] - z[dst]).pow(2).sum(dim=1)
    if alpha is None:
        return diff2.mean()
    a = alpha[mask]
    return (a * diff2).mean()


# ================================
# GNN variants
# ================================
# class FixedEdgeWeightGNN(nn.Module):
#     def __init__(self, dim, edge_type_emb_dim=8, hidden=128, dropout=args.gnn_dropout, trans_dropout=args.trans_dropout, edge_type_emb_dim=args.edge_type_emb_dim, edge_mlp_hidden=args.edge_mlp_hidden,
#                  use_edge_type=True, use_abs=True):
#         super().__init__()
#         self.use_edge_type = use_edge_type
#         self.use_abs = use_abs
#         self.edge_type_emb_dim = edge_type_emb_dim

#         self.type_emb = nn.Embedding(2, edge_type_emb_dim)
#         in_dim = dim * 2 + (dim if use_abs else 0) + (edge_type_emb_dim if use_edge_type else 0)

#         self.edge_mlp = nn.Sequential(
#             nn.Linear(in_dim, hidden),
#             nn.ReLU(),
#             nn.Linear(hidden, 1)
#         )
#         self.msg_mlp = nn.Sequential(
#             nn.Linear(dim, dim),
#             nn.ReLU(),
#             nn.Linear(dim, dim)
#         )
#         self.norm = nn.LayerNorm(dim)
#         self.dropout = dropout

    # def forward(self, z, edge_index, edge_type):
    #     src = edge_index[0]; dst = edge_index[1]
    #     zi = z[src]; zj = z[dst]

    #     parts = [zi, zj]
    #     if self.use_abs:
    #         parts.append((zi - zj).abs())

    #     if self.use_edge_type:
    #         te = self.type_emb(edge_type)
    #         parts.append(te)

    #     e_in = torch.cat(parts, dim=1)
    #     w = torch.sigmoid(self.edge_mlp(e_in)).squeeze(1)  # [E]

    #     # per-src softmax
    #     N = z.size(0)
    #     max_per_src = torch.full((N,), -1e9, device=z.device)
    #     max_per_src.scatter_reduce_(0, src, w, reduce="amax", include_self=True)
    #     w_exp = torch.exp(w - max_per_src[src])
    #     sum_per_src = torch.zeros((N,), device=z.device)
    #     sum_per_src.scatter_add_(0, src, w_exp)
    #     alpha = w_exp / (sum_per_src[src] + 1e-12)

    #     msg = self.msg_mlp(zj) * alpha.unsqueeze(1)
    #     agg = torch.zeros_like(z)
    #     agg.scatter_add_(0, src.unsqueeze(1).expand(-1, z.size(1)), msg)

    #     out = self.norm(z + F.dropout(agg, p=self.dropout, training=self.training))
    #     return out, alpha
    
class FixedEdgeWeightGNN(nn.Module):
    def __init__(
        self,
        dim,
        edge_type_emb_dim=8,
        hidden=128,
        dropout=0.1,
        use_edge_type=True,
        use_abs=True,
    ):
        super().__init__()
        self.use_edge_type = use_edge_type
        self.use_abs = use_abs
        self.edge_type_emb_dim = edge_type_emb_dim

        self.type_emb = nn.Embedding(2, edge_type_emb_dim)
        in_dim = dim * 2 + (dim if use_abs else 0) + (edge_type_emb_dim if use_edge_type else 0)

        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.msg_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = dropout

    def forward(self, z, edge_index, edge_type):
        src = edge_index[0]; dst = edge_index[1]
        zi = z[src]; zj = z[dst]

        parts = [zi, zj]
        if self.use_abs:
            parts.append((zi - zj).abs())

        if self.use_edge_type:
            te = self.type_emb(edge_type)
            parts.append(te)

        e_in = torch.cat(parts, dim=1)
        w = torch.sigmoid(self.edge_mlp(e_in)).squeeze(1)  # [E]

        # per-src softmax
        N = z.size(0)
        max_per_src = torch.full((N,), -1e9, device=z.device)
        max_per_src.scatter_reduce_(0, src, w, reduce="amax", include_self=True)
        w_exp = torch.exp(w - max_per_src[src])
        sum_per_src = torch.zeros((N,), device=z.device)
        sum_per_src.scatter_add_(0, src, w_exp)
        alpha = w_exp / (sum_per_src[src] + 1e-12)

        msg = self.msg_mlp(zj) * alpha.unsqueeze(1)
        agg = torch.zeros_like(z)
        agg.scatter_add_(0, src.unsqueeze(1).expand(-1, z.size(1)), msg)

        out = self.norm(z + F.dropout(agg, p=self.dropout, training=self.training))
        return out, alpha


class FixedUniformWeightGNN(nn.Module):
    """Uniform alpha=1/deg(src) (no edge_mlp)."""
    def __init__(self, dim, dropout=0.1, trans_dropout=trans_dropout, edge_type_emb_dim=8, edge_mlp_hidden=128):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = dropout

    def forward(self, z, edge_index, edge_type=None):
        src = edge_index[0]; dst = edge_index[1]
        zj = z[dst]
        N = z.size(0)

        deg = torch.zeros((N,), device=z.device, dtype=z.dtype)
        one = torch.ones((src.size(0),), device=z.device, dtype=z.dtype)
        deg.scatter_add_(0, src, one)

        alpha = 1.0 / (deg[src] + 1e-12)
        msg = self.msg_mlp(zj) * alpha.unsqueeze(1)

        agg = torch.zeros_like(z)
        agg.scatter_add_(0, src.unsqueeze(1).expand(-1, z.size(1)), msg)

        out = self.norm(z + F.dropout(agg, p=self.dropout, training=self.training))
        return out, alpha


class LossOnlyWeightGNN(nn.Module):
    """Learn alpha, but aggregate with uniform alpha; return learned alpha for loss weighting."""
    def __init__(self, dim, edge_type_emb_dim=8, hidden=128, dropout=0.1,
                 use_edge_type=True, use_abs=True):
        super().__init__()
        self.learn = FixedEdgeWeightGNN(dim, edge_type_emb_dim, hidden, dropout, use_edge_type, use_abs)
        self.uniform = FixedUniformWeightGNN(dim, dropout)

    def forward(self, z, edge_index, edge_type):
        # learned alpha (but ignore its agg)
        _, alpha_learned = self.learn(z, edge_index, edge_type)
        out, _ = self.uniform(z, edge_index, edge_type)
        return out, alpha_learned


# ================================
# Embedding models
# ================================
class EmbeddingModel(nn.Module):
    def __init__(self, input_dim, embedding_dim=256, hidden_dim=256,
                 num_heads=4, num_layers=1,
                 model_mode="full",
                 gnn_mode="learn",
                 use_edge_type=True,
                 use_abs=True,
                 dropout=0.1):
        super().__init__()
        self.model_mode = model_mode

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Transformer (optional)
        if model_mode in ("full", "transformer_only"):
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True, dropout=0.2
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        else:
            self.encoder = None
            # simple MLP for graph_only
            self.mlp = nn.Sequential(
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )

        self.out_proj = nn.Linear(hidden_dim, embedding_dim)

        # GNN (optional)
        self.gnn = None
        if model_mode in ("full", "graph_only"):
            if gnn_mode == "uniform":
                self.gnn = FixedUniformWeightGNN(dim=embedding_dim, dropout=dropout)
            elif gnn_mode == "loss_only":
                self.gnn = LossOnlyWeightGNN(dim=embedding_dim, edge_type_emb_dim=edge_type_emb_dim, hidden=edge_mlp_hidden, dropout=dropout,
                                             use_edge_type=use_edge_type, use_abs=use_abs)
            else:
                # gnn_mode == "learn" / "no_type" / "no_abs" etc are mapped by flags
                # self.gnn = FixedEdgeWeightGNN(dim=embedding_dim, edge_type_emb_dim=edge_type_emb_dim, hidden=edge_mlp_hidden, dropout=dropout,
                #                               use_edge_type=use_edge_type, use_abs=use_abs)
                
                self.gnn = FixedEdgeWeightGNN(
                    dim=embedding_dim,
                    edge_type_emb_dim=args.edge_type_emb_dim,
                    hidden=args.edge_mlp_hidden,
                    dropout=args.gnn_dropout,
                    use_edge_type=True,
                    use_abs=True,
                )

        self.dropout = dropout

    def forward(self, x_sub, edge_index, edge_type):
        h = self.input_proj(x_sub)

        if self.encoder is not None:
            h = h.unsqueeze(0)      # [1, Nsub, hidden]
            h = self.encoder(h).squeeze(0)
        else:
            h = self.mlp(h)

        z = self.out_proj(h)
        z = F.normalize(z, dim=1)

        if self.gnn is None:
            # dummy alpha to keep training loop stable
            E = edge_index.size(1)
            alpha = torch.ones((E,), device=z.device, dtype=z.dtype)
            return z, alpha

        z2, alpha = self.gnn(z, edge_index, edge_type)
        z2 = F.normalize(z2, dim=1)
        return z2, alpha


# ================================
# Inference
# ================================
@torch.no_grad()
def infer_all_embeddings(model, features,
                         neighbors_intra, neighbors_cross,
                         batch_size=512, n_expand_intra=8, n_expand_cross=3,
                         device="cuda"):
    model.eval()
    N = features.shape[0]
    D = model.out_proj.out_features
    Z = np.zeros((N, D), dtype=np.float32)
    features = np.asarray(features)

    for start in range(0, N, batch_size):
        seed_idx = np.arange(start, min(start + batch_size, N), dtype=np.int64)

        sub_set = set(seed_idx.tolist())
        for g in seed_idx:
            for nb in neighbors_intra[g][:n_expand_intra]:
                sub_set.add(nb)
            for nb in neighbors_cross[g][:n_expand_cross]:
                sub_set.add(nb)

        sub_nodes = np.array(sorted(list(sub_set)), dtype=np.int64)
        pos = {g: i for i, g in enumerate(sub_nodes)}
        seed_pos = np.array([pos[g] for g in seed_idx], dtype=np.int64)

        rows, cols, etype = [], [], []
        for g in sub_nodes:
            src = pos[g]
            for nb in neighbors_intra[g][:n_expand_intra]:
                if nb in pos:
                    rows.append(src); cols.append(pos[nb]); etype.append(0)
            for nb in neighbors_cross[g][:n_expand_cross]:
                if nb in pos:
                    rows.append(src); cols.append(pos[nb]); etype.append(1)

        edge_index = torch.tensor([rows, cols], dtype=torch.long, device=device)
        edge_type  = torch.tensor(etype, dtype=torch.long, device=device)
        x_sub = torch.tensor(features[sub_nodes], dtype=torch.float32, device=device)
        seed_pos_t = torch.tensor(seed_pos, dtype=torch.long, device=device)

        z_sub, _ = model(x_sub, edge_index, edge_type)
        Z[seed_idx] = z_sub[seed_pos_t].cpu().numpy()

    return Z


# ================================
# Main
# ================================
def parse_args():
    p = argparse.ArgumentParser("train_ablation")

    # IO
    p.add_argument("--input_h5ad", type=str, required=True)
    p.add_argument("--output_h5ad", type=str, required=True)

    # device
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=0)

    # preprocessing / graph
    p.add_argument("--label_key", type=str, default="labels3")   # training label
    p.add_argument("--species_key", type=str, default="species")
    p.add_argument("--pca_dim", type=int, default=50)
    p.add_argument("--k_intra", type=int, default=15)
    p.add_argument("--k_mnn", type=int, default=5)
    p.add_argument("--mnn_mode", type=str, default="by_label",
                   choices=["by_label", "global", "random_control"])

    # subgraph expansion / batching
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--n_expand_intra", type=int, default=8)
    p.add_argument("--n_expand_cross", type=int, default=3)

    # model
    p.add_argument("--embedding_dim", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=1)

    # dropout / gnn hyper-parameters (for sensitivity analysis)
    p.add_argument("--trans_dropout", type=float, default=0.2)
    p.add_argument("--gnn_dropout", type=float, default=0.1)
    p.add_argument("--edge_type_emb_dim", type=int, default=8)
    p.add_argument("--edge_mlp_hidden", type=int, default=128)

    p.add_argument("--model_mode", type=str, default="full",
                   choices=["full", "transformer_only", "graph_only"])

    # gnn
    p.add_argument("--gnn_mode", type=str, default="learn",
                   choices=["learn", "uniform", "loss_only"])
    p.add_argument("--no_edge_type", action="store_true")  # #7
    p.add_argument("--no_abs", action="store_true")        # #8

    # losses
    p.add_argument("--loss_mode", type=str, default="softtriple",
                   choices=["softtriple", "no_st", "ce", "softtriple_c1", "softtriple_frozen"])
    p.add_argument("--n_centers", type=int, default=2)
    # SoftTriple hyper-parameters (for sensitivity analysis)
    p.add_argument("--st_la", type=float, default=10.0)
    p.add_argument("--st_gamma", type=float, default=0.1)
    p.add_argument("--st_tau", type=float, default=0.2)


    p.add_argument("--lambda_var", type=float, default=0.2)
    p.add_argument("--lambda_graph", type=float, default=0.1)
    p.add_argument("--lambda_mnn", type=float, default=0.2)
    p.add_argument("--lambda_wreg", type=float, default=0.01)
    p.add_argument("--alpha_in_loss", action="store_true")  # if set: use alpha-weighted losses; else unweighted

    # training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=2025)

    # inference
    p.add_argument("--infer_batch", type=int, default=512)

    # umap export
    p.add_argument("--umap_dir", type=str, default="umap")
    p.add_argument("--umap_n_neighbors", type=int, default=12)
    p.add_argument("--umap_min_dist", type=float, default=0.7)

    # leiden (for UMAP visualization)
    p.add_argument("--leiden_resolution", type=float, default=1.0)
    p.add_argument("--leiden_key", type=str, default="leiden_transformer")

    # UMAP file naming
    p.add_argument("--run_name", type=str, default="")  # optional user-defined tag for output filenames

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---- load / preprocess ----
    adata = sc.read(args.input_h5ad)
    adata = adata[~adata.obs[args.label_key].isna() & ~adata.obs[args.species_key].isna()].copy()
    sc.pp.scale(adata)

    # encode labels/species
    label_enc = LabelEncoder()
    species_enc = LabelEncoder()
    adata.obs[f"{args.label_key}_enc"] = label_enc.fit_transform(adata.obs[args.label_key])
    adata.obs[f"{args.species_key}_enc"] = species_enc.fit_transform(adata.obs[args.species_key])

    labels = adata.obs[f"{args.label_key}_enc"].values.astype(np.int64)
    species_labels = adata.obs[f"{args.species_key}_enc"].values.astype(np.int64)
    n_classes = len(label_enc.classes_)

    features = to_dense_float32(adata.X)
    print("features:", features.shape, "labels:", labels.shape, "species:", species_labels.shape, "n_classes:", n_classes)

    # ---- graph build (PCA) ----
    pca = PCA(n_components=args.pca_dim, random_state=args.seed)
    X_graph = pca.fit_transform(features).astype(np.float32)

    neighbors_intra = build_intra_species_knn(X_graph, species_labels, k=args.k_intra)

    if args.mnn_mode == "by_label":
        neighbors_cross_real = build_cross_species_mnn_by_label(X_graph, labels, species_labels, k=args.k_mnn)
        neighbors_cross = neighbors_cross_real
    elif args.mnn_mode == "global":
        neighbors_cross_real = build_cross_species_mnn_global(X_graph, species_labels, k=args.k_mnn)
        neighbors_cross = neighbors_cross_real
    else:
        # random_control uses by_label as base by default
        neighbors_cross_real = build_cross_species_mnn_by_label(X_graph, labels, species_labels, k=args.k_mnn)
        neighbors_cross = randomize_cross_edges(neighbors_cross_real, species_labels, seed=args.seed)

    print("Graph built. Example intra:", neighbors_intra[0][:5], "cross:", neighbors_cross[0][:5])

    # ---- dataloader ----
    dataset = IdxDataset(len(features))
    collate_fn = make_subgraph_collate(
        features, labels, species_labels,
        neighbors_intra, neighbors_cross,
        n_expand_intra=args.n_expand_intra,
        n_expand_cross=args.n_expand_cross
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=False
    )

    # ---- model ----
    use_edge_type = (not args.no_edge_type)
    use_abs = (not args.no_abs)

    # map no_type/no_abs to learn-gnn with flags
    model = EmbeddingModel(
        input_dim=features.shape[1],
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        model_mode=args.model_mode,
        gnn_mode=args.gnn_mode,
        use_edge_type=use_edge_type,
        use_abs=use_abs,
        dropout=0.1
    ).to(device)

    # ---- loss / optimizer ----
    criterion = None
    cls_head = None

    n_centers = args.n_centers
    if args.loss_mode == "softtriple_c1":
        n_centers = 1

    if args.loss_mode in ("softtriple", "softtriple_c1", "softtriple_frozen"):
        criterion = SoftTripleLoss(args.embedding_dim, n_classes, n_centers=n_centers, la=args.st_la, gamma=args.st_gamma, tau=args.st_tau).to(device)
        if args.loss_mode == "softtriple_frozen":
            criterion.centers.requires_grad_(False)
            criterion.weight.requires_grad_(False)

    if args.loss_mode == "ce":
        cls_head = nn.Linear(args.embedding_dim, n_classes).to(device)

    # optimizer params
    params = list(model.parameters())
    if args.loss_mode == "ce":
        params += list(cls_head.parameters())
    if criterion is not None and args.loss_mode not in ("softtriple_frozen",):
        # full softtriple learns centers/weights
        params += list(criterion.parameters())

    optimizer = torch.optim.Adam(params, lr=args.lr)

    # ---- train ----
    print("\nStart training...")
    for epoch in range(args.epochs):
        model.train()
        running = 0.0

        for x_sub, y_seed, sp_seed, edge_index, edge_type, seed_pos in dataloader:
            x_sub = x_sub.to(device, non_blocking=True)
            y_seed = y_seed.to(device, non_blocking=True)
            sp_seed = sp_seed.to(device, non_blocking=True)
            edge_index = edge_index.to(device, non_blocking=True)
            edge_type = edge_type.to(device, non_blocking=True)
            seed_pos = seed_pos.to(device, non_blocking=True)

            z_sub, alpha = model(x_sub, edge_index, edge_type)
            z_seed = z_sub[seed_pos]

            # main supervised loss
            if args.loss_mode == "no_st":
                loss_st = torch.tensor(0.0, device=device)
            elif args.loss_mode == "ce":
                logits = cls_head(z_seed)
                loss_st = F.cross_entropy(logits, y_seed)
            else:
                loss_st = criterion(z_seed, y_seed)

            loss_v = variance_loss(z_seed, y_seed) if args.lambda_var != 0 else torch.tensor(0.0, device=device)

            if args.alpha_in_loss:
                loss_g = graph_smoothness_loss(z_sub, edge_index, edge_type, alpha=alpha)
                loss_p = mnn_pull_loss(z_sub, edge_index, edge_type, alpha=alpha)
            else:
                loss_g = graph_smoothness_loss(z_sub, edge_index, edge_type, alpha=None)
                loss_p = mnn_pull_loss(z_sub, edge_index, edge_type, alpha=None)

            loss_w = alpha.mean()

            loss = (loss_st
                    + args.lambda_var * loss_v
                    + args.lambda_graph * loss_g
                    + args.lambda_mnn * loss_p
                    + args.lambda_wreg * loss_w)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += loss.item() * y_seed.size(0)

        epoch_loss = running / len(dataset)
        print(f"Epoch [{epoch+1}/{args.epochs}] | Loss: {epoch_loss:.4f} "
              f"| ST: {float(loss_st.detach().cpu()):.4f} Var: {float(loss_v.detach().cpu()):.4f} "
              f"| Graph: {float(loss_g.detach().cpu()):.4f} MNN: {float(loss_p.detach().cpu()):.4f}")

    # ---- infer embeddings ----
    print("\nInfer all embeddings...")
    all_embeddings = infer_all_embeddings(
        model, features,
        neighbors_intra, neighbors_cross,
        batch_size=args.infer_batch,
        n_expand_intra=args.n_expand_intra,
        n_expand_cross=args.n_expand_cross,
        device=device
    )

    adata.obsm["X_embed_transformer"] = all_embeddings
    print("Saved embeddings:", all_embeddings.shape)

    # ---- UMAP plots (saved under ./umap, avoid overwrite via parameter-tagged run id) ----
    try:
        sc.pp.neighbors(adata, use_rep="X_embed_transformer", n_neighbors=args.umap_n_neighbors)
        sc.tl.umap(adata, min_dist=args.umap_min_dist, random_state=args.seed)

        # leiden for visualization
        if args.leiden_key not in adata.obs:
            sc.tl.leiden(adata, resolution=args.leiden_resolution, key_added=args.leiden_key, random_state=args.seed)

        # Save under CURRENT WORKING DIRECTORY
        umap_dir = os.path.join(os.getcwd(), args.umap_dir)
        os.makedirs(umap_dir, exist_ok=True)

        param_parts = [
            f"mode={args.model_mode}",
            f"gnn={args.gnn_mode}",
            f"loss={args.loss_mode}",
            f"nc={n_centers}",
            f"lv={args.lambda_var}",
            f"lg={args.lambda_graph}",
            f"lm={args.lambda_mnn}",
            f"lw={args.lambda_wreg}",
            f"L={args.num_layers}",
            f"H={args.num_heads}",
        ]
        if args.run_name:
            param_parts.append(f"tag={args.run_name}")
        param_tag = "_".join(param_parts).replace("/", "-")

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        base = os.path.splitext(os.path.basename(args.output_h5ad))[0]

        plots = [
            (args.species_key, "species"),
            (args.label_key, "labels2"),
            (args.leiden_key, "leiden"),
        ]
        for color_key, suffix in plots:
            if color_key in adata.obs:
                sc.pl.umap(adata, color=color_key, show=False)
                out_png = os.path.join(umap_dir, f"{base}_{param_tag}_{run_id}_{suffix}.png")
                plt.savefig(out_png, dpi=200, bbox_inches="tight")
                plt.close()

        print("Saved UMAP PNGs to:", umap_dir)
    except Exception as e:
        print("[WARN] UMAP plotting failed:", repr(e))

    # ---- save ----
    os.makedirs(os.path.dirname(args.output_h5ad), exist_ok=True)
    adata.write(args.output_h5ad)
    print("Wrote:", args.output_h5ad)


if __name__ == "__main__":
    main()
