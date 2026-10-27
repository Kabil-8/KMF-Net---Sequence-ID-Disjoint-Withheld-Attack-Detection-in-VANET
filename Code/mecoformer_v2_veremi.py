#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MeCoFormer-v2: multi-scale temporal + cross-feature attention + benign memory
for VeReMi sequence anomaly/misbehaviour detection.

Kaggle:
  !pip install -q optuna
  !python mecoformer_v2_veremi.py --smoke
  !python mecoformer_v2_veremi.py
  !python mecoformer_v2_veremi.py --models mecoformer mecoformer_nomem mecoformer_nofeat lstm transformer

Expected layout:
  ROOT/sequences/{train,validation,test,open_set_attack}_*.npz
  each NPZ contains X [N,T,F], y_multiclass [N], optional sequence_uid and y_3class.
  ROOT may contain sequence_manifest.csv or sequence_manifest.csv.gz.

Important protocol:
- Feature scaler is fitted on the training split only.
- Hyperparameters are tuned on a fixed stratified training subset using validation.
- The chosen hyperparameters are frozen for the final three-seed runs.
- Test and open-set labels are never used for model selection or threshold selection.
- The open_set_attack split is evaluation-only.
"""
import argparse
import copy
import glob
import itertools
import json
import math
import os
import random
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, matthews_corrcoef,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    optuna = None

SPLITS = ("train", "validation", "test", "open_set_attack")
SEEDS_DEFAULT = (42, 2026, 7)


def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Reproducibility is preferred over maximum speed for the reported runs.
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        pass


def find_root(path=""):
    candidates = []
    if path:
        candidates.append(path)
        candidates.extend(os.path.dirname(p) for p in glob.glob(os.path.join(path, "**", "sequences"), recursive=True))
    if os.path.isdir("/kaggle/input"):
        candidates.extend(os.path.dirname(p) for p in glob.glob("/kaggle/input/**/sequences", recursive=True))
    for root in candidates:
        if os.path.isdir(os.path.join(root, "sequences")) and glob.glob(os.path.join(root, "sequences", "train_*.npz")):
            return root
    raise FileNotFoundError("Cannot find ROOT/sequences/train_*.npz. Attach the dataset or pass --data ROOT.")


def find_manifest(root):
    for name in ("sequence_manifest.csv", "sequence_manifest.csv.gz"):
        path = os.path.join(root, name)
        if os.path.isfile(path):
            return path
    hits = glob.glob(os.path.join(root, "**", "sequence_manifest.csv*",), recursive=True)
    return hits[0] if hits else None


def npz_first(z, *names):
    for name in names:
        if name in z.files:
            return z[name]
    return None


class DatasetBundle:
    pass


def load_data(root, max_shards=None):
    raw = {}
    for split in SPLITS:
        files = sorted(glob.glob(os.path.join(root, "sequences", f"{split}_*.npz")))
        if max_shards is not None:
            files = files[:max_shards]
        if not files:
            raise FileNotFoundError(f"No {split}_*.npz files found under {root}/sequences")
        xs, ys, uids, y3s = [], [], [], []
        for path in files:
            with np.load(path, allow_pickle=True) as z:
                x = np.asarray(z["X"], dtype=np.float32)
                y = np.asarray(z["y_multiclass"], dtype=np.int64)
                if x.ndim != 3 or len(x) != len(y):
                    raise ValueError(f"Bad X/y shapes in {path}: X={x.shape}, y={y.shape}")
                xs.append(x)
                ys.append(y)
                uid = npz_first(z, "sequence_uid")
                if uid is None:
                    uid = np.asarray([f"{os.path.basename(path)}:{i}" for i in range(len(y))])
                uids.append(np.asarray(uid).astype(str))
                y3 = npz_first(z, "y_3class")
                y3s.append(None if y3 is None else np.asarray(y3, dtype=np.int64))
        raw[split] = {
            "X": np.concatenate(xs), "y": np.concatenate(ys),
            "uid": np.concatenate(uids),
            "y3": None if any(v is None for v in y3s) else np.concatenate(y3s),
        }
        log(f"loaded {split:16s} X={raw[split]['X'].shape}, labels={len(raw[split]['y'])}")

    # Check split independence where stable sequence identifiers are supplied.
    uid_sets = {s: set(raw[s]["uid"].tolist()) for s in SPLITS}
    for a, b in itertools.combinations(SPLITS, 2):
        overlap = uid_sets[a] & uid_sets[b]
        if overlap:
            raise ValueError(f"Sequence UID leakage between {a} and {b}: {len(overlap)} shared IDs")

    # Fit feature selection and scaling on TRAIN ONLY.
    xtr = raw["train"]["X"]
    if any(raw[s]["X"].shape[1:] != xtr.shape[1:] for s in SPLITS):
        raise ValueError("All splits must have identical T and F dimensions.")
    T, F0 = xtr.shape[1:]
    flat = xtr.reshape(-1, F0).astype(np.float64)
    mean0 = flat.mean(axis=0)
    std0 = flat.std(axis=0)
    keep = np.flatnonzero(std0 > 1e-6)
    if len(keep) == 0:
        raise ValueError("All features are constant on the training split.")
    mean = mean0[keep].astype(np.float32)
    std = np.maximum(std0[keep], 1e-6).astype(np.float32)
    log("kept feature indices:", keep.tolist())

    known_raw = sorted(set(raw["train"]["y"].tolist()) |
                       set(raw["validation"]["y"].tolist()) |
                       set(raw["test"]["y"].tolist()))
    if not known_raw or 0 not in known_raw:
        raise ValueError(f"Expected benign raw label 0 in known splits; found {known_raw}")
    class_map = {c: i for i, c in enumerate(known_raw)}
    n_cls = len(known_raw)

    # Establish raw-label -> hierarchical three-class map. Require consistency.
    raw_to_3 = {}
    for split in ("train", "validation", "test"):
        y3 = raw[split]["y3"]
        if y3 is not None:
            for c, c3 in zip(raw[split]["y"], y3):
                c, c3 = int(c), int(c3)
                if c in raw_to_3 and raw_to_3[c] != c3:
                    raise ValueError(f"Inconsistent y_3class mapping for raw class {c}")
                raw_to_3[c] = c3
    if not all(c in raw_to_3 for c in known_raw):
        mp = find_manifest(root)
        if mp:
            cols = pd.read_csv(mp, nrows=0).columns
            if {"y_multiclass", "y_3class"}.issubset(cols):
                man = pd.read_csv(mp, usecols=["y_multiclass", "y_3class"])
                for c, grp in man.groupby("y_multiclass"):
                    vals = grp["y_3class"].dropna().astype(int).unique()
                    if len(vals) == 1:
                        raw_to_3[int(c)] = int(vals[0])
    if not all(c in raw_to_3 for c in known_raw):
        raise ValueError(
            "A complete raw y_multiclass -> y_3class mapping is required. "
            "Provide y_3class in shards or in sequence_manifest.csv; do not silently substitute binary labels."
        )
    c3_values = sorted(set(raw_to_3[c] for c in known_raw))
    if not set(c3_values).issubset({0, 1, 2}):
        raise ValueError(f"Expected y_3class labels in {{0,1,2}}, found {c3_values}")

    M3 = np.zeros((n_cls, 3), dtype=np.float32)
    for raw_label, idx in class_map.items():
        M3[idx, raw_to_3[raw_label]] = 1.0

    D = DatasetBundle()
    D.keep = keep
    D.scaler_mean = mean
    D.scaler_std = std
    D.known_raw = known_raw
    D.class_map = class_map
    D.M3 = torch.from_numpy(M3)
    D.n_cls, D.T, D.F = n_cls, T, len(keep)
    for split, tag in zip(SPLITS, ("tr", "va", "te", "op")):
        x = (raw[split]["X"][..., keep] - mean[None, None, :]) / std[None, None, :]
        # Bound extreme values without using validation/test statistics.
        x = np.clip(x, -12.0, 12.0).astype(np.float32)
        raw_y = raw[split]["y"]
        ymc = np.asarray([class_map.get(int(c), -1) for c in raw_y], dtype=np.int64)
        ybin = (raw_y != 0).astype(np.float32)
        y3 = np.asarray([raw_to_3.get(int(c), 2) for c in raw_y], dtype=np.int64)
        setattr(D, "X" + tag, torch.from_numpy(np.ascontiguousarray(x)))
        setattr(D, "mc" + tag, torch.from_numpy(ymc).long())
        setattr(D, "b" + tag, torch.from_numpy(ybin).float())
        setattr(D, "c3" + tag, torch.from_numpy(y3).long())
        setattr(D, "uid" + tag, raw[split]["uid"])
    log(f"T={D.T}, F={D.F}, known classes={D.n_cls}, raw labels={known_raw}")
    return D


def move_tensors(D, device):
    for key, value in list(D.__dict__.items()):
        if torch.is_tensor(value):
            setattr(D, key, value.to(device))
    return D


# ------------------------------- architecture -------------------------------
class KinematicStem(nn.Module):
    """Input, first differences and second differences with stable normalization."""
    def __init__(self, n_features):
        super().__init__()
        self.out_channels = n_features * 3
        self.norm = nn.LayerNorm(self.out_channels)

    def forward(self, x):
        d1 = torch.cat((torch.zeros_like(x[:, :1]), x[:, 1:] - x[:, :-1]), dim=1)
        d2 = torch.cat((torch.zeros_like(d1[:, :1]), d1[:, 1:] - d1[:, :-1]), dim=1)
        return self.norm(torch.cat((x, d1, d2), dim=-1))


class TemporalConvBlock(nn.Module):
    def __init__(self, channels, kernel, dilation, dropout):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 1),
        )
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x):
        return x + self.norm(self.net(x))


class EncoderLayer(nn.Module):
    def __init__(self, d, heads, ff_mult, dropout):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, d * ff_mult), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d * ff_mult, d), nn.Dropout(dropout)
        )

    def forward(self, x):
        q = self.attn_norm(x)
        a, _ = self.attn(q, q, q, need_weights=False)
        x = x + a
        return x + self.ff(self.ff_norm(x))


class ScaleEncoder(nn.Module):
    def __init__(self, in_channels, T, d, patch, heads, layers, ff_mult, dropout):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv1d(in_channels, d, kernel_size=patch, stride=patch, padding=0)
        n_tokens = T // patch
        if n_tokens < 1:
            raise ValueError(f"Patch {patch} exceeds sequence length {T}")
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.layers = nn.ModuleList([EncoderLayer(d, heads, ff_mult, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        # x: B,T,C; pad right to a multiple of patch size.
        xt = x.transpose(1, 2)
        remainder = xt.shape[-1] % self.patch
        if remainder:
            xt = F.pad(xt, (0, self.patch - remainder), mode="replicate")
        z = self.proj(xt).transpose(1, 2)
        if z.shape[1] != self.pos.shape[1]:
            # This branch supports T not divisible by patch; interpolate position embeddings.
            pos = F.interpolate(self.pos.transpose(1, 2), size=z.shape[1], mode="linear", align_corners=False).transpose(1, 2)
        else:
            pos = self.pos
        z = z + pos
        for layer in self.layers:
            z = layer(z)
        return self.norm(z)


class MultiScaleTemporal(nn.Module):
    def __init__(self, channels, T, d, patches, heads, layers, ff_mult, dropout):
        super().__init__()
        self.conv = nn.Sequential(
            TemporalConvBlock(channels, 3, 1, dropout),
            TemporalConvBlock(channels, 5, 2, dropout),
            TemporalConvBlock(channels, 3, 4, dropout),
        )
        self.scales = nn.ModuleList([
            ScaleEncoder(channels, T, d, p, heads, layers, ff_mult, dropout) for p in patches
        ])
        self.cross = nn.ModuleList([
            nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True) for _ in patches
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in patches])
        self.out = nn.Sequential(nn.Linear(d * len(patches), d), nn.GELU(), nn.LayerNorm(d))

    def forward(self, x):
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        toks = [enc(x) for enc in self.scales]
        pooled = []
        for i, t in enumerate(toks):
            others = [toks[j] for j in range(len(toks)) if j != i]
            if others:
                context = torch.cat(others, dim=1)
                a, _ = self.cross[i](t, context, context, need_weights=False)
                t = self.norms[i](t + a)
            pooled.append(t.mean(dim=1))
        return self.out(torch.cat(pooled, dim=-1))


class CrossFeatureEncoder(nn.Module):
    """Each feature's temporal trace is a token; attention learns feature interactions."""
    def __init__(self, channels, T, d, heads, layers, ff_mult, dropout):
        super().__init__()
        self.embed = nn.Linear(T, d)
        self.channel_embed = nn.Parameter(torch.zeros(1, channels, d))
        nn.init.trunc_normal_(self.channel_embed, std=0.02)
        self.layers = nn.ModuleList([EncoderLayer(d, heads, ff_mult, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        z = self.embed(x.transpose(1, 2)) + self.channel_embed
        for layer in self.layers:
            z = layer(z)
        return self.norm(z).mean(dim=1)


class BenignMemory(nn.Module):
    def __init__(self, slots, d, topk, temperature=0.08):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(slots, d) * 0.02)
        self.topk = min(topk, slots)
        self.temperature = temperature

    def forward(self, z):
        logits = F.normalize(z.float(), dim=-1) @ F.normalize(self.memory.float(), dim=-1).T
        weights = (logits / self.temperature).softmax(dim=-1)
        if self.topk < weights.shape[-1]:
            ids = weights.topk(self.topk, dim=-1).indices
            mask = torch.zeros_like(weights).scatter_(1, ids, 1.0)
            weights = weights * mask
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        recon = weights @ self.memory.float()
        entropy = -(weights * weights.clamp_min(1e-8).log()).sum(-1)
        return recon, weights, entropy


class PredictionHeads(nn.Module):
    def __init__(self, in_dim, hidden, n_cls, dropout):
        super().__init__()
        self.body = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden),
                                  nn.GELU(), nn.Dropout(dropout))
        self.multiclass = nn.Linear(hidden, n_cls)
        self.binary = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return self.multiclass(h), self.binary(h).squeeze(-1)


def focal_ce(logits, y, gamma):
    logp = F.log_softmax(logits.float(), dim=-1).gather(1, y[:, None]).squeeze(1)
    p = logp.exp()
    return (-((1.0 - p).clamp_min(0) ** gamma) * logp).mean()


def focal_bce(logits, y, gamma):
    bce = F.binary_cross_entropy_with_logits(logits.float(), y.float(), reduction="none")
    return (((1.0 - torch.exp(-bce)).clamp_min(0) ** gamma) * bce).mean()


def supervised_contrastive(features, labels, temperature=0.1):
    features = F.normalize(features.float(), dim=-1)
    b = len(labels)
    if b < 2:
        return features.sum() * 0
    eye = torch.eye(b, dtype=torch.bool, device=features.device)
    sim = (features @ features.T) / temperature
    sim = sim.masked_fill(eye, -1e9)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    positives = (labels[:, None] == labels[None, :]) & ~eye
    counts = positives.sum(dim=1)
    valid = counts > 0
    if not valid.any():
        return features.sum() * 0
    # Avoid 0 * -inf / NaN by indexing positive entries per anchor.
    pos_logp = (log_prob * positives.float()).sum(dim=1)
    return -(pos_logp[valid] / counts[valid]).mean()


class MeCoFormerV2(nn.Module):
    def __init__(self, meta, hp, ablation="none"):
        super().__init__()
        self.ablation = ablation
        F0, T, n_cls = meta["F"], meta["T"], meta["n_cls"]
        d, heads = hp["d_model"], hp["heads"]
        self.n_attack = n_cls - 1
        self.n_proto = hp["n_proto"]
        self.T, self.F0 = T, F0
        self.register_buffer("M3", meta["M3"].clone())
        self.stem = KinematicStem(F0)
        c = self.stem.out_channels
        self.temporal = MultiScaleTemporal(c, T, d, hp["patches"], heads, hp["layers"], hp["ff_mult"], hp["dropout"])
        self.feature = CrossFeatureEncoder(c, T, d, heads, hp["layers"], hp["ff_mult"], hp["dropout"])
        self.gate = nn.Sequential(nn.Linear(2*d, d), nn.GELU(), nn.Linear(d, d))
        self.fuse_norm = nn.LayerNorm(d)
        self.memory = BenignMemory(hp["mem_slots"], d, hp["topk"])
        self.decoder = nn.Sequential(nn.Linear(d, 2*d), nn.GELU(), nn.Dropout(hp["dropout"]), nn.Linear(2*d, T*F0))
        self.attack_prototypes = nn.Parameter(torch.randn(max(1, self.n_attack * self.n_proto), d) * 0.02)
        self.projection = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 64))
        self.heads = PredictionHeads(2*d + self.n_attack, d, n_cls, hp["dropout"])

    def forward(self, x):
        b = x.shape[0]
        s = self.stem(x)
        zt = self.temporal(s)
        zf = self.feature(s)
        if self.ablation == "nofeat":
            gate = torch.ones_like(zt)
            z = self.fuse_norm(zt)
        elif self.ablation == "notime":
            gate = torch.zeros_like(zt)
            z = self.fuse_norm(zf)
        else:
            gate = torch.sigmoid(self.gate(torch.cat((zt, zf), dim=-1)))
            z = self.fuse_norm(gate * zt + (1.0 - gate) * zf)

        zhat, weights, entropy = self.memory(z)
        reconstruction = self.decoder(zhat).view(b, self.T, self.F0)
        rec_error = (reconstruction.float() - x.float()).square().mean(dim=(1, 2))
        residual_vec = z.float() - zhat.detach().float()
        residual = residual_vec.norm(dim=-1)

        sims = F.normalize(z.float(), dim=-1) @ F.normalize(self.attack_prototypes.float(), dim=-1).T
        if self.n_attack:
            sims_grouped = sims[:, :self.n_attack*self.n_proto].reshape(b, self.n_attack, self.n_proto)
            class_similarity = sims_grouped.max(dim=-1).values
        else:
            class_similarity = z.new_zeros((b, 0))

        if self.ablation in ("nomem", "nonormalmem"):
            residual_vec = torch.zeros_like(residual_vec)
        if self.ablation in ("nomem", "noattackmem"):
            class_similarity = torch.zeros_like(class_similarity)
        mc, binary = self.heads(torch.cat((z.float(), residual_vec, class_similarity), dim=-1))
        return {
            "mc": mc, "bin": binary, "res": residual, "rec": rec_error,
            "msim": class_similarity.max(dim=-1).values if class_similarity.shape[1] else z.new_zeros(b),
            "gate": gate.mean(dim=-1), "z": z, "zhat": zhat,
            "ent": entropy, "sims_c": class_similarity,
            "proj": F.normalize(self.projection(z).float(), dim=-1),
        }

    def compute_loss(self, out, y, hp):
        logits, labels = out["mc"].float(), y["mc"]
        loss = focal_ce(logits, labels, hp["gamma"]) + focal_bce(out["bin"], y["bin"], hp["gamma"])
        p3 = (logits.softmax(dim=-1) @ self.M3).clamp_min(1e-8)
        loss = loss + hp["lam_hier"] * F.nll_loss(p3.log(), y["c3"])
        loss = loss + hp["lam_con"] * supervised_contrastive(out["proj"], labels)
        benign = labels == 0
        if self.ablation not in ("nomem", "nonormalmem") and benign.any():
            # Memory/reconstruction objectives are applied only to benign training examples.
            loss = loss + hp["lam_rec"] * out["rec"][benign].mean()
            loss = loss + hp["lam_feat"] * F.mse_loss(out["zhat"][benign].float(), out["z"][benign].float().detach())
            loss = loss + hp["lam_ent"] * out["ent"][benign].mean()
        if self.ablation not in ("nomem", "noattackmem") and self.n_attack:
            attack = ~benign
            if attack.any():
                # Known class IDs are mapped to 0..n_cls-1, benign is 0.
                # Each attack class c uses prototype group c-1.
                loss = loss + hp["lam_proto"] * F.cross_entropy(out["sims_c"][attack] / hp["proto_temp"], labels[attack] - 1)
            if benign.any():
                loss = loss + hp["lam_proto"] * F.relu(out["sims_c"][benign].max(dim=-1).values - hp["benign_proto_margin"]).mean()
        return loss


class Baseline(nn.Module):
    def __init__(self, meta, hp, kind):
        super().__init__()
        self.stem = KinematicStem(meta["F"])
        c, d, T = self.stem.out_channels, hp["d_model"], meta["T"]
        if kind == "lstm":
            h = hp["hidden"]
            self.encoder = nn.LSTM(c, h, hp["layers"], batch_first=True, bidirectional=hp["bidir"],
                                   dropout=hp["dropout"] if hp["layers"] > 1 else 0)
            od = h * (2 if hp["bidir"] else 1)
            self.head = PredictionHeads(2*od, h, meta["n_cls"], hp["dropout"])
            self.kind = kind
        elif kind == "cnnlstm":
            channels, h = hp["channels"], hp["hidden"]
            self.conv = nn.Sequential(nn.Conv1d(c, channels, 3, padding=1), nn.GELU(),
                                      nn.Conv1d(channels, channels, 5, padding=2), nn.GELU())
            self.encoder = nn.LSTM(channels, h, hp["layers"], batch_first=True,
                                   dropout=hp["dropout"] if hp["layers"] > 1 else 0)
            self.head = PredictionHeads(2*h, h, meta["n_cls"], hp["dropout"])
            self.kind = kind
        elif kind == "transformer":
            self.input = nn.Linear(c, d)
            self.pos = nn.Parameter(torch.zeros(1, T, d))
            nn.init.trunc_normal_(self.pos, std=0.02)
            self.layers = nn.ModuleList([EncoderLayer(d, hp["heads"], hp["ff_mult"], hp["dropout"]) for _ in range(hp["layers"])])
            self.norm = nn.LayerNorm(d)
            self.head = PredictionHeads(d, d, meta["n_cls"], hp["dropout"])
            self.kind = kind
        elif kind == "itransformer":
            self.encoder = CrossFeatureEncoder(c, T, d, hp["heads"], hp["layers"], hp["ff_mult"], hp["dropout"])
            self.head = PredictionHeads(d, d, meta["n_cls"], hp["dropout"])
            self.kind = kind
        else:
            raise ValueError(f"Unknown model {kind}")

    def forward(self, x):
        s = self.stem(x)
        if self.kind == "lstm":
            o, _ = self.encoder(s)
            h = torch.cat((o[:, -1], o.mean(1)), dim=-1)
        elif self.kind == "cnnlstm":
            c = self.conv(s.transpose(1, 2)).transpose(1, 2)
            o, _ = self.encoder(c)
            h = torch.cat((o[:, -1], o.mean(1)), dim=-1)
        elif self.kind == "transformer":
            o = self.input(s) + self.pos
            for layer in self.layers:
                o = layer(o)
            h = self.norm(o).mean(1)
        else:
            h = self.encoder(s)
        mc, binary = self.head(h)
        return {"mc": mc, "bin": binary}

    def compute_loss(self, out, y, hp):
        return focal_ce(out["mc"], y["mc"], hp["gamma"]) + focal_bce(out["bin"], y["bin"], hp["gamma"])


COMMON = dict(lr=8e-4, wd=1e-3, bs=256, dropout=0.15, gamma=1.0, noise=0.01,
              sampler_power=0.5, ema=0.995, lam_hier=0.5, lam_con=0.1,
              lam_rec=0.5, lam_ent=0.005, lam_proto=0.3, lam_feat=0.1,
              proto_temp=0.1, benign_proto_margin=0.2)
DEFAULTS = {
    "mecoformer": dict(COMMON, d_model=96, heads=4, layers=2, ff_mult=2, patches=[2,4,8],
                       mem_slots=128, topk=8, n_proto=2),
    "lstm": dict(COMMON, hidden=128, layers=2, bidir=True, d_model=96, heads=4, ff_mult=2),
    "cnnlstm": dict(COMMON, channels=64, hidden=128, layers=1, d_model=96, heads=4, ff_mult=2),
    "transformer": dict(COMMON, d_model=96, heads=4, layers=2, ff_mult=2),
    "itransformer": dict(COMMON, d_model=96, heads=4, layers=2, ff_mult=2),
}


def build_model(name, hp, meta):
    if name.startswith("mecoformer"):
        suffix = name[len("mecoformer"):].lstrip("_")
        ablation = suffix if suffix else "none"
        return MeCoFormerV2(meta, hp, ablation)
    return Baseline(meta, hp, name)


def suggest(trial, name, T):
    hp = dict(
        lr=trial.suggest_float("lr", 2e-4, 2e-3, log=True),
        wd=trial.suggest_float("wd", 1e-5, 5e-2, log=True),
        bs=trial.suggest_categorical("bs", [128, 256, 512]),
        dropout=trial.suggest_float("dropout", 0.0, 0.3),
        gamma=trial.suggest_categorical("gamma", [0.0, 1.0, 2.0]),
        noise=trial.suggest_categorical("noise", [0.0, 0.005, 0.01, 0.02]),
        sampler_power=trial.suggest_categorical("sampler_power", [0.0, 0.25, 0.5, 0.75]),
        ema=trial.suggest_categorical("ema", [0.99, 0.995, 0.999]),
        lam_hier=trial.suggest_float("lam_hier", 0.1, 1.0),
        lam_con=trial.suggest_float("lam_con", 0.0, 0.3),
    )
    if name == "mecoformer":
        valid_patches = ["-".join(map(str, p)) for p in ([2,4,8], [2,4], [4,8], [2,8], [1,2,4,8]) if max(p) <= T]
        if not valid_patches:
            valid_patches = ["1"]
        hp.update(
            d_model=trial.suggest_categorical("d_model", [64, 96, 128]),
            heads=trial.suggest_categorical("heads", [4, 8]),
            layers=trial.suggest_int("layers", 1, 3),
            ff_mult=trial.suggest_categorical("ff_mult", [2, 4]),
            patches=trial.suggest_categorical("patches", valid_patches),
            mem_slots=trial.suggest_categorical("mem_slots", [64, 128, 256]),
            topk=trial.suggest_categorical("topk", [4, 8, 16]),
            n_proto=trial.suggest_int("n_proto", 1, 3),
            lam_rec=trial.suggest_float("lam_rec", 0.05, 1.5, log=True),
            lam_ent=trial.suggest_float("lam_ent", 1e-4, 0.05, log=True),
            lam_proto=trial.suggest_float("lam_proto", 0.05, 1.0, log=True),
            lam_feat=trial.suggest_float("lam_feat", 0.01, 0.5, log=True),
        )
    elif name == "lstm":
        hp.update(hidden=trial.suggest_categorical("hidden", [64,128,256]),
                  layers=trial.suggest_int("layers", 1, 3),
                  bidir=trial.suggest_categorical("bidir", [True, False]),
                  d_model=96, heads=4, ff_mult=2)
    elif name == "cnnlstm":
        hp.update(channels=trial.suggest_categorical("channels", [32,64,128]),
                  hidden=trial.suggest_categorical("hidden", [64,128,256]),
                  layers=trial.suggest_int("layers", 1, 2),
                  d_model=96, heads=4, ff_mult=2)
    else:
        hp.update(d_model=trial.suggest_categorical("d_model", [64,96,128]),
                  heads=trial.suggest_categorical("heads", [4,8]),
                  layers=trial.suggest_int("layers", 1, 4),
                  ff_mult=trial.suggest_categorical("ff_mult", [2,4]))
    # Merge sampled values over the complete model defaults. This preserves
    # fixed loss parameters (e.g. proto_temp and benign_proto_margin) that are
    # consumed by compute_loss but are intentionally not Optuna search fields.
    hp = {**DEFAULTS["mecoformer" if name.startswith("mecoformer") else name], **hp}
    if name == "mecoformer" and isinstance(hp.get("patches"), str):
        hp["patches"] = [int(v) for v in hp["patches"].split("-")]
    return hp


def selection_score(pred, y_mc, y_bin, n_cls):
    ap = average_precision_score(y_bin, pred["bin"])
    _, f1 = best_f1_threshold(y_bin, pred["bin"])
    mf = f1_score(y_mc, pred["mc"].argmax(1), average="macro",
                  labels=list(range(n_cls)), zero_division=0)
    return 0.4 * ap + 0.3 * f1 + 0.3 * mf


def best_f1_threshold(y, score):
    p, r, thresholds = precision_recall_curve(y, score)
    if len(thresholds) == 0:
        return 0.0, 0.0
    f1 = 2*p[:-1]*r[:-1] / (p[:-1] + r[:-1] + 1e-12)
    idx = int(np.nanargmax(f1))
    return float(thresholds[idx]), float(f1[idx])


KEEP = ("mc", "bin", "res", "rec", "msim", "gate")


@torch.no_grad()
def predict(model, X, device, batch_size=1024):
    model.eval()
    outputs = defaultdict(list)
    amp = device.type == "cuda"
    for start in range(0, len(X), batch_size):
        xb = X[start:start+batch_size]
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            out = model(xb)
        for key in KEEP:
            if key in out:
                outputs[key].append(out[key].float().cpu())
    return {k: np.nan_to_num(torch.cat(v).numpy(), nan=0.0, posinf=1e4, neginf=-1e4)
            for k, v in outputs.items()}


def fit(name, hp, D, seed, epochs, patience, device, train_indices=None, on_epoch=None, verbose=False):
    seed_everything(seed)
    meta = {"F": D.F, "T": D.T, "n_cls": D.n_cls, "M3": D.M3}
    model = build_model(name, hp, meta).to(device)
    ema_model = copy.deepcopy(model).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    all_idx = torch.arange(len(D.Xtr), device=device) if train_indices is None else torch.as_tensor(train_indices, device=device)
    labels_all = D.mctr[all_idx]
    if (labels_all < 0).any():
        raise ValueError("Unknown/open-set labels must not occur in the training split.")
    counts = torch.bincount(labels_all, minlength=D.n_cls).float().clamp_min(1)
    sample_weights = (1.0 / counts[labels_all]).pow(hp["sampler_power"])
    N, bs = len(all_idx), int(hp["bs"])
    steps = max(1, math.ceil(N / bs))
    total_steps = max(1, epochs * steps)
    warmup = max(1, int(0.03 * total_steps))
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: (step+1)/warmup if step < warmup else
        0.5*(1+math.cos(math.pi*min(1.0, (step-warmup)/max(1,total_steps-warmup))))
    )
    amp = device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp)

    def ema_update():
        with torch.no_grad():
            for ep, p in zip(ema_model.parameters(), model.parameters()):
                ep.mul_(hp["ema"]).add_(p.detach(), alpha=1-hp["ema"])
            for eb, b in zip(ema_model.buffers(), model.buffers()):
                eb.copy_(b)

    yva_mc = D.mcva.detach().cpu().numpy()
    yva_bin = D.bva.detach().cpu().numpy()
    best, best_state, bad, history = -float("inf"), None, 0, []
    for epoch in range(epochs):
        model.train()
        start_time, total_loss = time.time(), 0.0
        if hp["sampler_power"] > 0:
            draw = all_idx[torch.multinomial(sample_weights, len(all_idx), replacement=True)]
        else:
            draw = all_idx[torch.randperm(len(all_idx), device=device)]
        for j in range(steps):
            batch_idx = draw[j*bs:(j+1)*bs]
            if len(batch_idx) == 0:
                continue
            xb = D.Xtr[batch_idx]
            if hp["noise"] > 0:
                xb = xb + hp["noise"] * torch.randn_like(xb)
            y = {"mc": D.mctr[batch_idx], "bin": D.btr[batch_idx], "c3": D.c3tr[batch_idx]}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                out = model(xb)
            loss = model.compute_loss(out, y, hp)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch+1}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema_update()
            total_loss += float(loss.detach())
        val_pred = predict(ema_model, D.Xva, device)
        score = selection_score(val_pred, yva_mc, yva_bin, D.n_cls)
        history.append(score)
        if verbose:
            log(f"  epoch {epoch+1:02d}/{epochs} loss={total_loss/max(1,steps):.4f} val_select={score:.4f} ({time.time()-start_time:.0f}s)")
        if score > best:
            best, bad = score, 0
            best_state = copy.deepcopy(ema_model.state_dict())
        else:
            bad += 1
        if on_epoch is not None:
            on_epoch(epoch, score)
        if bad >= patience:
            break
    if best_state is not None:
        ema_model.load_state_dict(best_state)
    return ema_model, best, history


def tune(name, D, device, trials, tune_epochs, tune_fraction, out_dir, seed=42):
    base = "mecoformer" if name.startswith("mecoformer") else name
    path = os.path.join(out_dir, f"tuning_{base}.json")
    if os.path.isfile(path):
        saved = json.load(open(path, encoding="utf-8"))
        log(f"[{base}] reusing frozen hyperparameters from {path}")
        params = saved["best_params"]
        if isinstance(params.get("patches"), str):
            params["patches"] = [int(v) for v in params["patches"].split("-")]
        return params
    if trials <= 0:
        return dict(DEFAULTS[base])
    if optuna is None:
        raise ImportError("Optuna is required for tuning. Run: pip install -q optuna")
    idx = np.arange(len(D.Xtr))
    if tune_fraction < 1:
        try:
            idx, _ = train_test_split(idx, train_size=tune_fraction,
                                      stratify=D.mctr.detach().cpu().numpy(), random_state=seed)
        except ValueError:
            idx, _ = train_test_split(idx, train_size=tune_fraction, random_state=seed)

    def objective(trial):
        hp = suggest(trial, base, D.T)
        if hp["d_model"] % hp["heads"] != 0:
            raise optuna.TrialPruned()
        def report(epoch, score):
            trial.report(score, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        try:
            _, score, _ = fit(base, hp, D, seed, tune_epochs, patience=3,
                              device=device, train_indices=idx, on_epoch=report)
            return score
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" not in str(exc).lower() and not isinstance(exc, torch.cuda.OutOfMemoryError):
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise optuna.TrialPruned()

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=min(4, max(1, trials//2)), n_warmup_steps=2))
    study.optimize(objective, n_trials=trials, gc_after_trial=True)
    hp = dict(DEFAULTS[base])
    hp.update(study.best_params)
    if isinstance(hp.get("patches"), str):
        hp["patches"] = [int(v) for v in hp["patches"].split("-")]
    elif hp.get("patches") is None:
        hp["patches"] = [2,4,8]
    json.dump({"best_value": float(study.best_value), "best_params": hp,
               "selection_protocol": "train subset + validation; frozen before final seeds"},
              open(path, "w", encoding="utf-8"), indent=2)
    study.trials_dataframe().to_csv(os.path.join(out_dir, f"trials_{base}.csv"), index=False)
    log(f"[{base}] best validation selection score={study.best_value:.4f}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return hp


def fusion_stats(pred):
    keys = [k for k in ("res", "rec", "msim") if k in pred]
    stats = {k: (float(np.mean(pred[k])), float(np.std(pred[k]) + 1e-6)) for k in ("bin", *keys)}
    return stats


def fusion_score(pred, stats, weights):
    score = (pred["bin"] - stats["bin"][0]) / stats["bin"][1]
    for key, weight in zip(("res", "rec", "msim"), weights):
        if weight and key in pred and key in stats:
            score = score + weight * (pred[key] - stats[key][0]) / stats[key][1]
    return score


def tune_fusion(pred_val, y_val):
    stats = fusion_stats(pred_val)
    best_ap, best_weights = -1.0, (0.0, 0.0, 0.0)
    for weights in itertools.product((0.0, 0.25, 0.5, 1.0), repeat=3):
        score = fusion_score(pred_val, stats, weights)
        ap = average_precision_score(y_val, score)
        if ap > best_ap:
            best_ap, best_weights = ap, weights
    return stats, best_weights


def evaluate(pred, y_mc, y_bin, y3, score, threshold, D):
    binary_hat = (score >= threshold).astype(int)
    mc_hat = pred["mc"].argmax(axis=1)
    p3 = softmax_np(pred["mc"]) @ D.M3.detach().cpu().numpy()
    c3_hat = p3.argmax(axis=1)
    result = {
        "bin_acc": accuracy_score(y_bin, binary_hat),
        "bin_prec": precision_score(y_bin, binary_hat, zero_division=0),
        "bin_rec": recall_score(y_bin, binary_hat, zero_division=0),
        "bin_f1": f1_score(y_bin, binary_hat, zero_division=0),
        "bin_mcc": matthews_corrcoef(y_bin, binary_hat),
        "bin_auroc": roc_auc_score(y_bin, score),
        "bin_auprc": average_precision_score(y_bin, score),
        "mc_acc": accuracy_score(y_mc, mc_hat),
        "mc_macro_f1": f1_score(y_mc, mc_hat, average="macro", labels=list(range(D.n_cls)), zero_division=0),
        "mc_weighted_f1": f1_score(y_mc, mc_hat, average="weighted", labels=list(range(D.n_cls)), zero_division=0),
        "c3_acc": accuracy_score(y3, c3_hat),
        "c3_macro_f1": f1_score(y3, c3_hat, average="macro", labels=[0,1,2], zero_division=0),
    }
    return result


def softmax_np(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)


def open_set_metrics(benign_scores, unseen_scores, threshold):
    y = np.r_[np.zeros(len(benign_scores)), np.ones(len(unseen_scores))]
    scores = np.r_[benign_scores, unseen_scores]
    threshold_95 = float(np.percentile(unseen_scores, 5))
    return {
        "open_auroc": roc_auc_score(y, scores),
        "open_auprc": average_precision_score(y, scores),
        "open_fpr_at_95_unknown_tpr": float((benign_scores >= threshold_95).mean()),
        "open_detection_rate_at_val_threshold": float((unseen_scores >= threshold).mean()),
    }


def run_seed(name, hp, D, seed, epochs, patience, device, out_dir):
    model, best_score, history = fit(name, hp, D, seed, epochs, patience, device, verbose=True)
    pval, ptest, popen = (predict(model, X, device) for X in (D.Xva, D.Xte, D.Xop))
    yval_bin = D.bva.detach().cpu().numpy()
    ytest_bin = D.bte.detach().cpu().numpy()
    ytest_mc = D.mcte.detach().cpu().numpy()
    ytest_3 = D.c3te.detach().cpu().numpy()
    if (D.mcva < 0).any() or (D.mcte < 0).any():
        raise ValueError("Known validation/test splits contain labels outside the known class map.")
    variants = {"head": None}
    if name.startswith("mecoformer") and name not in ("mecoformer_nomem",):
        variants["fused"] = "fused"
    results = {}
    for variant, mode in variants.items():
        if mode is None:
            score_fn = lambda p: p["bin"]
            stats, weights = None, (0.0, 0.0, 0.0)
        else:
            stats, weights = tune_fusion(pval, yval_bin)
            score_fn = lambda p, st=stats, w=weights: fusion_score(p, st, w)
        val_score = score_fn(pval)
        threshold, val_f1 = best_f1_threshold(yval_bin, val_score)
        test_score = score_fn(ptest)
        open_score = score_fn(popen)
        metrics = evaluate(ptest, ytest_mc, ytest_bin, ytest_3, test_score, threshold, D)
        benign_test_scores = test_score[ytest_bin == 0]
        metrics.update(open_set_metrics(benign_test_scores, open_score, threshold))
        metrics.update({
            "val_selection_score": best_score,
            "val_f1_at_selected_threshold": val_f1,
            "epochs_run": len(history),
            "threshold_from_validation": threshold,
        })
        if "gate" in ptest:
            metrics["temporal_gate_mean":] = float(ptest["gate"].mean())
        if stats is not None:
            metrics["fusion_weights_res_rec_msim"] = str(weights)
        results[variant] = metrics
    torch.save({"model": model.state_dict(), "model_name": name, "seed": seed, "hp": hp,
                "class_map": D.class_map, "kept_features": D.keep,
                "scaler_mean": D.scaler_mean, "scaler_std": D.scaler_std},
               os.path.join(out_dir, f"{name}_seed{seed}.pt"))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


METRICS = [
    "bin_f1", "bin_auprc", "bin_auroc", "bin_mcc", "bin_acc", "bin_prec", "bin_rec",
    "mc_macro_f1", "mc_weighted_f1", "c3_macro_f1", "open_auroc", "open_auprc",
    "open_detection_rate_at_val_threshold", "open_fpr_at_95_unknown_tpr",
]


def summarize(frame):
    grouped = frame.groupby(["model", "variant"])
    means = grouped[METRICS].mean(numeric_only=True)
    stds = grouped[METRICS].std(ddof=1, numeric_only=True).fillna(0)
    pretty = means.copy().astype(object)
    for metric in METRICS:
        pretty[metric] = [f"{a:.4f} ± {b:.4f}" for a, b in zip(means[metric], stds[metric])]
    return pretty.reset_index(), means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="")
    parser.add_argument("--out", default="/kaggle/working/runs/mecoformer_v2")
    parser.add_argument("--models", nargs="+", default=["mecoformer", "lstm", "cnnlstm", "transformer", "itransformer"])
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--baseline_trials", type=int, default=6)
    parser.add_argument("--tune_epochs", type=int, default=6)
    parser.add_argument("--tune_fraction", type=float, default=0.30)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS_DEFAULT))
    parser.add_argument("--smoke", action="store_true")
    args, _ = parser.parse_known_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log("device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "(enable GPU in Kaggle settings)")
    if args.smoke:
        args.trials, args.baseline_trials = 1, 1
        args.tune_epochs, args.epochs, args.patience = 1, 2, 2
        args.seeds = args.seeds[:1]
        args.out = args.out.rstrip("/") + "_smoke"
        os.makedirs(args.out, exist_ok=True)

    root = find_root(args.data)
    log("data root:", root)
    D = move_tensors(load_data(root, max_shards=2 if args.smoke else None), device)
    results_path = os.path.join(args.out, "results_per_seed.csv")
    records = pd.read_csv(results_path).to_dict("records") if os.path.isfile(results_path) else []
    done = {(r["model"], int(r["seed"]), r["variant"]) for r in records}
    tuned = {}

    for name in args.models:
        base = "mecoformer" if name.startswith("mecoformer") else name
        missing = [s for s in args.seeds if not all((name, s, v) in done for v in
                   (["head", "fused"] if name.startswith("mecoformer") and name != "mecoformer_nomem" else ["head"]))]
        if not missing:
            log(f"[{name}] all requested seeds complete; skipping")
            continue
        if base not in tuned:
            t0 = time.time()
            n_trials = args.trials if base == "mecoformer" else args.baseline_trials
            tuned[base] = tune(base, D, device, n_trials, args.tune_epochs,
                               args.tune_fraction, args.out, seed=args.seeds[0])
            log(f"[{base}] tuning finished in {(time.time()-t0)/60:.1f} min")
        hp = tuned[base]
        for seed in missing:
            log(f"=== {name} | seed={seed} ===")
            outputs = run_seed(name, hp, D, seed, args.epochs, args.patience, device, args.out)
            for variant, metrics in outputs.items():
                records.append({"model": name, "variant": variant, "seed": seed, **metrics})
            pd.DataFrame(records).to_csv(results_path, index=False)
            json.dump(tuned, open(os.path.join(args.out, "best_hparams.json"), "w", encoding="utf-8"), indent=2)

    frame = pd.DataFrame(records)
    if frame.empty:
        raise RuntimeError("No results produced.")
    pretty, means = summarize(frame)
    pretty.to_csv(os.path.join(args.out, "summary_mean_std.csv"), index=False)
    try:
        report = pretty.to_markdown(index=False)
    except Exception:
        report = pretty.to_string(index=False)
    open(os.path.join(args.out, "summary_mean_std.md"), "w", encoding="utf-8").write(report)
    print("\n" + report)
    print("\nBest model per metric (mean across completed seeds):")
    for metric in METRICS:
        idx = means[metric].idxmin() if metric == "open_fpr_at_95_unknown_tpr" else means[metric].idxmax()
        print(f"  {metric:38s} {idx} {means.loc[idx, metric]:.4f}")
    log("Outputs:", args.out)


if __name__ == "__main__":
    main()
