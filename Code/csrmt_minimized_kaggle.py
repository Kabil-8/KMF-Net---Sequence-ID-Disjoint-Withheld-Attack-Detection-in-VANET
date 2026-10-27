#!/usr/bin/env python3
"""
CSRMT: Cross-Scale Relational Memory Transformer.

Includes:
  - Full-feature Transformer (128 dimensions, 4 heads, 3 layers)
  - Temporal dilated CNN with kernels 3/5 and dilations 1/2/4
  - Learned vehicle-relation attention pathway
  - Dynamic three-way cross-scale gating
  - Momentum-updated class prototype memory
  - Known-class and unknown-confidence heads
  - Validation-only random hyperparameter search
  - Three-seed training and early stopping
  - Ablation experiments
  - Closed-set and open-set evaluation
  - Mean +/- standard deviation reporting

Expected data:
  DATA_ROOT/
      sequences/
          train_*.npz
          validation_*.npz
          test_*.npz
          open_set_attack_*.npz

Each NPZ contains:
  X: [number_of_sequences, sequence_length, number_of_features]
  y_multiclass: [number_of_sequences]

IMPORTANT:
Only the minimized dataset path is accepted. Normalization is fit on training
shards only. Hyperparameter selection and early stopping use validation data.
The unknown threshold is calibrated on known validation examples. Test data
is used only for final evaluation. No Mamba dependency is required.
"""

import argparse
import gc
import json
import math
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import IterableDataset, DataLoader

from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


SEEDS = [42, 52, 62]
SPLITS = ["train", "validation", "test", "open_set_attack"]


# ============================================================
# 1. Reproducibility and data loading
# ============================================================

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_split_files(root, split):
    files = sorted((root / "sequences").glob(f"{split}_*.npz"))
    if not files:
        raise FileNotFoundError(
            f"No {split}_*.npz files found in {root / 'sequences'}"
        )
    return files


def load_label_vocabulary(train_files):
    labels = set()
    for file in train_files:
        with np.load(file, allow_pickle=False) as z:
            labels.update(
                int(v) for v in np.unique(z["y_multiclass"])
            )

    if not labels:
        raise ValueError("No training labels found.")

    ordered = sorted(labels)
    return {old: new for new, old in enumerate(ordered)}


def infer_dimensions(train_files):
    with np.load(train_files[0], allow_pickle=False) as z:
        x = z["X"]
        if x.ndim != 3:
            raise ValueError(
                f"Expected X with shape [N,T,F], got {x.shape}"
            )
        return int(x.shape[1]), int(x.shape[2])


def compute_train_feature_stats(train_files, n_features):
    """Compute per-feature mean/std from TRAIN shards only."""
    sums = np.zeros(n_features, dtype=np.float64)
    sums_sq = np.zeros(n_features, dtype=np.float64)
    counts = np.zeros(n_features, dtype=np.int64)

    for file in train_files:
        with np.load(file, allow_pickle=False) as z:
            x = np.asarray(z["X"], dtype=np.float64)
            if x.ndim != 3 or x.shape[-1] != n_features:
                raise ValueError(f"Unexpected training tensor shape in {file}: {x.shape}")
            for f in range(n_features):
                values = x[:, :, f].reshape(-1)
                valid = np.isfinite(values)
                values = values[valid]
                if values.size:
                    sums[f] += values.sum()
                    sums_sq[f] += np.square(values).sum()
                    counts[f] += values.size

    mean = np.divide(sums, np.maximum(counts, 1))
    variance = np.divide(sums_sq, np.maximum(counts, 1)) - mean ** 2
    std = np.sqrt(np.maximum(variance, 0.0))
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


class SequenceShardDataset(IterableDataset):
    """Stream NPZ shards; shuffle files and sequences for training."""

    def __init__(
        self, files, label_map, shuffle=False, seed=42,
        feature_mean=None, feature_std=None
    ):
        super().__init__()
        self.files = list(files)
        self.label_map = label_map
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.feature_mean = (
            None if feature_mean is None else np.asarray(feature_mean, dtype=np.float32)
        )
        self.feature_std = (
            None if feature_std is None else np.asarray(feature_std, dtype=np.float32)
        )

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        files = self.files.copy()

        if self.shuffle:
            rng.shuffle(files)

        for file in files:
            with np.load(file, allow_pickle=False) as z:
                x = z["X"]
                y = z["y_multiclass"]

                indices = np.arange(len(y))
                if self.shuffle:
                    rng.shuffle(indices)

                for i in indices:
                    xi = np.asarray(x[i], dtype=np.float32)
                    yi = self.label_map.get(int(y[i]), -1)

                    # Keep numerical failures from poisoning training.
                    xi = np.nan_to_num(
                        xi,
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                    if self.feature_mean is not None and self.feature_std is not None:
                        xi = (xi - self.feature_mean) / self.feature_std
                    yield torch.from_numpy(np.ascontiguousarray(xi)), int(yi)


def make_loader(
    files, label_map, batch_size, shuffle, seed, workers=0,
    feature_mean=None, feature_std=None
):
    ds = SequenceShardDataset(
        files, label_map, shuffle=shuffle, seed=seed,
        feature_mean=feature_mean, feature_std=feature_std
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


# ============================================================
# 2. Train-only feature selection
# ============================================================

def sequence_summaries(x):
    """
    x: [N,T,F]
    Returns four sequence-level descriptors per feature:
    mean, standard deviation, range, and end-to-start change.
    """
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    mean = x.mean(axis=1)
    std = x.std(axis=1)
    span = x.max(axis=1) - x.min(axis=1)
    change = x[:, -1, :] - x[:, 0, :]

    return np.concatenate(
        [mean, std, span, change], axis=1
    )


def fisher_scores(X, y):
    """Multiclass Fisher score for every summary column."""
    classes = np.unique(y)
    overall = X.mean(axis=0)
    between = np.zeros(X.shape[1])
    within = np.zeros(X.shape[1])

    for c in classes:
        xc = X[y == c]
        if len(xc) == 0:
            continue
        between += len(xc) * (xc.mean(axis=0) - overall) ** 2
        within += ((xc - xc.mean(axis=0)) ** 2).sum(axis=0)

    return between / (within + 1e-12)


def average_rank_score(scores):
    """Convert scores to [0,1] ranks without scale assumptions."""
    order = np.argsort(np.argsort(scores, kind="stable"), kind="stable")
    if len(scores) <= 1:
        return np.ones_like(scores, dtype=float)
    return order / (len(scores) - 1)


def select_features(
    train_files, label_map, n_features, method="hybrid",
    max_sequences=200000, seed=42
):
    """
    IMPORTANT: call this with training files only.

    Ranking is performed on sequence-level descriptors, then
    importance is aggregated across descriptors belonging to
    the same original input feature.
    """
    summaries = []
    labels = []
    seen = 0

    for file in train_files:
        with np.load(file, allow_pickle=False) as z:
            x = z["X"]
            y = z["y_multiclass"]

            mapped_y = np.asarray(
                [label_map.get(int(v), -1) for v in y], dtype=np.int64
            )
            valid_idx = np.flatnonzero(mapped_y >= 0)
            remaining = max_sequences - seen
            if remaining <= 0:
                break
            valid_idx = valid_idx[:remaining]
            if len(valid_idx):
                summaries.append(sequence_summaries(x[valid_idx]))
                labels.append(mapped_y[valid_idx])
                seen += len(valid_idx)

        if seen >= max_sequences:
            break

    if not summaries:
        raise ValueError("No labelled training sequences available for feature selection.")
    Xs = np.concatenate(summaries, axis=0).astype(np.float64, copy=False)
    ys = np.concatenate(labels, axis=0).astype(np.int64, copy=False)

    if len(np.unique(ys)) < 2:
        raise ValueError(
            "Feature selection needs at least two training classes."
        )

    # Avoid MI failures from constant or invalid columns.
    med = np.nanmedian(Xs, axis=0)
    med = np.nan_to_num(med, nan=0.0)
    Xs = np.where(np.isfinite(Xs), Xs, med)

    # Standardize descriptor columns for numerical stability.
    mu = Xs.mean(axis=0)
    sd = Xs.std(axis=0)
    Xs = (Xs - mu) / np.maximum(sd, 1e-8)

    fs = fisher_scores(Xs, ys)

    # Mutual information may be expensive on very large data.
    mi = mutual_info_classif(
        Xs.astype(np.float32),
        ys,
        random_state=seed,
    )

    # 4 descriptors per original input feature.
    nf = Xs.shape[1] // 4
    fs_rank = average_rank_score(fs)
    mi_rank = average_rank_score(mi)

    fs_feature = fs_rank.reshape(4, nf).mean(axis=0)
    mi_feature = mi_rank.reshape(4, nf).mean(axis=0)

    if method == "fisher":
        importance = fs_feature
    elif method == "mi":
        importance = mi_feature
    elif method == "hybrid":
        importance = 0.5 * fs_feature + 0.5 * mi_feature
    else:
        raise ValueError("method must be fisher, mi, or hybrid")

    # Return the FULL ranking in descending importance order.
    # The caller chooses the top-k subset; tuning can therefore compare
    # smaller and larger subsets without losing highly ranked features.
    ranked = np.argsort(importance)[::-1]
    k = min(max(1, int(n_features)), nf)

    report = pd.DataFrame({
        "feature_index": np.arange(nf),
        "fisher_rank_score": fs_feature,
        "mi_rank_score": mi_feature,
        "combined_score": importance,
        "selected": np.isin(np.arange(nf), ranked[:k]),
    }).sort_values("combined_score", ascending=False)

    return ranked, report


# ============================================================
# 3. Model architecture
# ============================================================

class AttentionPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.score = nn.Linear(d, 1)

    def forward(self, x):
        weights = torch.softmax(self.score(x), dim=1)
        return (weights * x).sum(dim=1)


class TemporalDilatedCNN(nn.Module):
    """Six temporal scales: kernels 3/5 crossed with dilations 1/2/4."""
    def __init__(self, n_features, d=128, dropout=0.2):
        super().__init__()
        self.branches = nn.ModuleList()
        for kernel in (3, 5):
            for dilation in (1, 2, 4):
                padding = dilation * (kernel - 1) // 2
                self.branches.append(nn.Sequential(
                    nn.Conv1d(n_features, d, kernel_size=kernel,
                              dilation=dilation, padding=padding),
                    nn.BatchNorm1d(d), nn.GELU(),
                ))
        self.fuse = nn.Sequential(
            nn.Conv1d(6 * d, d, kernel_size=1), nn.GELU(), nn.Dropout(dropout)
        )

    def forward(self, x):
        # [B,T,F] -> [B,F,T] -> [B,T,D]
        z = x.transpose(1, 2)
        return self.fuse(torch.cat([b(z) for b in self.branches], dim=1)).transpose(1, 2)


class VehicleRelationAttention(nn.Module):
    """Cross-feature and temporal relation attention over learned feature roles.

    The supplied tensor contract specifies 20 columns but does not include a
    verified mapping from column indices to position/speed/acceleration/sender
    consistency. The pathway therefore learns feature-role embeddings and
    cross-feature interactions from values plus first/second temporal differences,
    instead of silently assuming undocumented column indices.
    """
    def __init__(self, n_features, d=128, heads=4, dropout=0.2):
        super().__init__()
        self.feature_id = nn.Parameter(torch.randn(1, 1, n_features, d) * 0.02)
        self.value_proj = nn.Linear(1, d)
        self.delta_proj = nn.Linear(1, d)
        self.accel_proj = nn.Linear(1, d)
        self.role_gate = nn.Sequential(
            nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 3), nn.Softmax(-1)
        )
        self.feature_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.feature_norm = nn.LayerNorm(d)
        self.temporal_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.temporal_norm = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))

    def forward(self, x):
        # Scalar feature tokens: [B,T,F,1] -> [B,T,F,D].
        delta = torch.diff(x, dim=1, prepend=x[:, :1, :])
        accel = torch.diff(delta, dim=1, prepend=delta[:, :1, :])
        xv, xd, xa = x.unsqueeze(-1), delta.unsqueeze(-1), accel.unsqueeze(-1)
        v = self.value_proj(xv) + self.feature_id
        d1 = self.delta_proj(xd) + self.feature_id
        d2 = self.accel_proj(xa) + self.feature_id
        weights = self.role_gate(torch.cat([v, d1, d2], dim=-1))
        tokens = weights[..., 0:1] * v + weights[..., 1:2] * d1 + weights[..., 2:3] * d2
        b, t, f, d = tokens.shape
        tokens = tokens.reshape(b * t, f, d)
        attended, _ = self.feature_attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.feature_norm(tokens + attended)
        # Aggregate cross-feature relations into one representation per timestep.
        temporal = tokens.mean(dim=1).reshape(b, t, d)
        temporal_attended, _ = self.temporal_attn(temporal, temporal, temporal, need_weights=False)
        temporal = self.temporal_norm(temporal + temporal_attended)
        return self.temporal_norm(temporal + self.ff(temporal))


class CSRMT(nn.Module):
    """Cross-Scale Relational Memory Transformer and ablation variants."""
    def __init__(self, n_features, n_classes, seq_len, d=128, heads=4,
                 layers=3, ff=256, dropout=0.2, temperature=0.1,
                 memory_momentum=0.9, variant="full_csrmt"):
        super().__init__()
        self.variant = variant
        self.temperature = max(float(temperature), 1e-3)
        self.memory_momentum = memory_momentum
        self.input_proj = nn.Linear(n_features, d)
        self.position = nn.Parameter(torch.zeros(1, seq_len, d))
        nn.init.trunc_normal_(self.position, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=ff, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=layers)

        self.temporal_cnn = None if variant in {"transformer_only", "no_temporal_cnn"} else TemporalDilatedCNN(n_features, d, dropout)
        self.relation = None if variant in {"transformer_only", "no_relation"} else VehicleRelationAttention(n_features, d, heads, dropout)
        active_scales = 1 + int(self.temporal_cnn is not None) + int(self.relation is not None)
        self.gate = None if variant in {"transformer_only", "no_gate"} else nn.Sequential(
            nn.Linear(active_scales * d, d), nn.GELU(), nn.Linear(d, active_scales), nn.Softmax(dim=-1))
        self.concat_fuse = nn.Sequential(nn.Linear(active_scales * d, d), nn.GELU(), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(d)
        self.pool = AttentionPool(d)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d, n_classes)
        self.unknown_head = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d // 2, 1))

        if variant in {"full_csrmt", "no_relation", "no_temporal_cnn", "no_gate"}:
            self.register_buffer("prototypes", torch.zeros(n_classes, d))
            self.register_buffer("prototype_seen", torch.zeros(n_classes, dtype=torch.bool))
            self.memory_classifier = nn.Linear(d, n_classes)
            self.proto_logit_scale = nn.Parameter(torch.tensor(0.1))
        else:
            self.register_buffer("prototypes", torch.zeros(n_classes, d))
            self.register_buffer("prototype_seen", torch.zeros(n_classes, dtype=torch.bool))
            self.memory_classifier = None
            self.register_parameter("proto_logit_scale", None)

    def encode(self, x):
        if x.size(1) > self.position.size(1):
            raise ValueError(f"Input sequence length {x.size(1)} exceeds configured maximum {self.position.size(1)}")
        scales = [self.transformer(self.input_proj(x) + self.position[:, :x.size(1)])]
        if self.temporal_cnn is not None:
            scales.append(self.temporal_cnn(x))
        if self.relation is not None:
            scales.append(self.relation(x))
        if self.gate is not None:
            weights = self.gate(torch.cat(scales, dim=-1))
            fused = sum(weights[..., i:i+1] * z for i, z in enumerate(scales))
        else:
            fused = self.concat_fuse(torch.cat(scales, dim=-1))
        return self.dropout(self.pool(self.norm(fused)))

    def update_prototypes(self, embeddings, labels):
        if self.memory_classifier is None:
            return
        with torch.no_grad():
            for cls in labels.unique():
                c = int(cls.item())
                mean = nn.functional.normalize(embeddings[labels == cls].mean(0), dim=0)
                if self.prototype_seen[c]:
                    updated = self.memory_momentum * self.prototypes[c] + (1 - self.memory_momentum) * mean
                    self.prototypes[c].copy_(nn.functional.normalize(updated, dim=0))
                else:
                    self.prototypes[c].copy_(mean)
                    self.prototype_seen[c] = True

    def forward(self, x, return_aux=False, return_embedding=False):
        h = self.encode(x)
        logits = self.classifier(h)
        proto_logits = None
        if self.memory_classifier is not None:
            p = nn.functional.normalize(self.prototypes, dim=-1)
            memory_logits = self.memory_classifier(h)
            proto_logits = (nn.functional.normalize(h, dim=-1) @ p.T) / self.temperature
            proto_logits = proto_logits.masked_fill(~self.prototype_seen.unsqueeze(0), -20.0)
            logits = logits + memory_logits + torch.tanh(self.proto_logit_scale) * proto_logits
        unknown_logit = self.unknown_head(h).squeeze(-1)
        if return_embedding:
            return h
        if return_aux:
            return logits, proto_logits, unknown_logit, h
        return logits


# ============================================================
# 4. Training, prediction and validation
# ============================================================

# ============================================================
# 4. Training, prediction and validation
# ============================================================

def make_model(nf, indices, nc, seq_len, cfg, variant):
    return CSRMT(
        n_features=nf, n_classes=nc, seq_len=seq_len, d=128,
        heads=4, layers=3, ff=256, dropout=cfg["dropout"],
        temperature=cfg["temperature"], variant=variant,
    )


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, probs, energies = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits, _, unknown_logit, _ = model(x, return_aux=True)
        p = torch.softmax(logits, dim=-1)
        ys.append(y.numpy())
        probs.append(p.cpu().numpy())
        energies.append(unknown_logit.cpu().numpy())
    return np.concatenate(ys), np.concatenate(probs), np.concatenate(energies)


def train_model(
    model, train_loader, val_loader, device,
    cfg, max_epochs, patience, checkpoint, seed,
):
    seed_all(seed)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_score = -float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss, total_n = 0.0, 0

        for x, y in train_loader:
            valid = y >= 0
            if not valid.any():
                continue

            x = x[valid].to(device, non_blocking=True)
            y = y[valid].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits, proto_logits, unknown_logits, embeddings = model(x, return_aux=True)
            loss = nn.functional.cross_entropy(logits, y, label_smoothing=0.02)
            if proto_logits is not None and model.memory_classifier is not None and model.prototype_seen.any():
                seen_targets = model.prototype_seen[y]
                if seen_targets.any():
                    loss = loss + cfg["proto_weight"] * nn.functional.cross_entropy(
                        proto_logits[seen_targets], y[seen_targets]
                    )

            # Train the unknown head without using open-set shards: original
            # training samples are known; feature-shuffled/noisy copies are pseudo-unknown.
            known_target = torch.zeros_like(unknown_logits)
            known_loss = nn.functional.binary_cross_entropy_with_logits(unknown_logits, known_target)
            perm = torch.randperm(x.size(0), device=x.device)
            pseudo_unknown = x[perm].clone()
            # Blend temporal order disruption and modest feature perturbation.
            if x.size(1) > 1:
                pseudo_unknown = torch.flip(pseudo_unknown, dims=[1])
            scale = pseudo_unknown.std(dim=(1, 2), keepdim=True).clamp_min(1e-3)
            pseudo_unknown = pseudo_unknown + 0.5 * scale * torch.randn_like(pseudo_unknown)
            _, _, pseudo_unknown_logits, _ = model(pseudo_unknown, return_aux=True)
            unknown_target = torch.ones_like(pseudo_unknown_logits)
            unknown_loss = 0.5 * (known_loss + nn.functional.binary_cross_entropy_with_logits(pseudo_unknown_logits, unknown_target))
            loss = loss + cfg.get("unknown_weight", 0.2) * unknown_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.update_prototypes(embeddings.detach(), y)

            total_loss += loss.item() * len(y)
            total_n += len(y)

        vy, vp, _ = predict(model, val_loader, device)
        valid = vy >= 0
        vy, vp = vy[valid], vp[valid]

        if len(vy) == 0:
            raise ValueError("Validation set has no known-class labels.")

        score = f1_score(
            vy, vp.argmax(axis=1), average="macro", zero_division=0
        )
        val_loss = -np.mean(
            np.log(np.clip(vp[np.arange(len(vy)), vy], 1e-12, 1))
        )

        scheduler.step(score)
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / max(total_n, 1),
            "val_loss": float(val_loss),
            "val_macro_f1": float(score),
            "lr": optimizer.param_groups[0]["lr"],
        })

        print(
            f"seed={seed} epoch={epoch:03d} "
            f"val_macro_f1={score:.5f} val_loss={val_loss:.5f}",
            flush=True,
        )

        if score > best_score + 1e-5:
            best_score = score
            best_epoch = epoch
            bad_epochs = 0
            torch.save({
                "state": model.state_dict(),
                "best_epoch": epoch,
                "best_val_macro_f1": float(score),
                "config": cfg,
            }, checkpoint)
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            break

    saved = torch.load(
        checkpoint, map_location=device, weights_only=False
    )
    model.load_state_dict(saved["state"])
    return model, history, saved


def classification_metrics(y, probs):
    valid = y >= 0
    y, probs = y[valid], probs[valid]
    pred = probs.argmax(axis=1)

    out = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(
            y, pred, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            y, pred, average="weighted", zero_division=0
        ),
        "mcc": matthews_corrcoef(y, pred),
    }

    try:
        out["macro_ovr_auc"] = roc_auc_score(
            y, probs, multi_class="ovr", average="macro"
        )
    except ValueError:
        out["macro_ovr_auc"] = np.nan

    return out


# ============================================================
# 5. Hyperparameter tuning
# ============================================================

def tune_config(
    root, label_map, indices, nf, nc, seq_len,
    args, device, output_dir, feature_mean, feature_std,
):
    """
    Random search using validation macro-F1 only.

    The chosen configuration is reused across ablations so that
    architecture comparisons share the same tuning budget.
    """
    rng = random.Random(args.seed)
    candidates = []

    for k in [nf]:
        candidate_indices = np.arange(nf, dtype=np.int64)
        for lr in [3e-4, 5e-4]:
            for dropout in [0.1, 0.2, 0.3]:
                candidates.append({
                    "top_k": len(candidate_indices),
                    "indices": candidate_indices.tolist(),
                    "lr": lr,
                    "weight_decay": 1e-4,
                    "dropout": dropout,
                    "temperature": 0.1,
                    "proto_weight": 0.1,
                    "unknown_weight": 0.2,
                })

    # Avoid a prohibitively large grid on a large dataset.
    rng.shuffle(candidates)
    candidates = candidates[:max(1, args.tune_trials)]
    if not candidates:
        raise ValueError("No hyperparameter candidates were generated.")

    train_files = get_split_files(root, "train")
    val_files = get_split_files(root, "validation")
    train_loader = make_loader(
        train_files, label_map, args.batch_size, True, args.seed,
        feature_mean=feature_mean, feature_std=feature_std
    )
    val_loader = make_loader(
        val_files, label_map, args.batch_size, False, args.seed,
        feature_mean=feature_mean, feature_std=feature_std
    )

    results = []
    best_cfg, best_score = None, -float("inf")

    for trial, cfg in enumerate(candidates, 1):
        idx = np.asarray(cfg.pop("indices"), dtype=np.int64)
        print(f"\nTUNING trial {trial}/{len(candidates)}: {cfg}")

        seed_all(args.seed)
        model = make_model(
            nf, idx, nc, seq_len, cfg, "full_csrmt"
        )
        ckpt = output_dir / f"tune_trial_{trial}.pt"

        model, hist, saved = train_model(
            model, train_loader, val_loader, device,
            cfg, args.tune_epochs, args.tune_patience,
            ckpt, args.seed,
        )

        score = saved["best_val_macro_f1"]
        results.append({
            **cfg,
            "val_macro_f1": score,
            "best_epoch": saved["best_epoch"],
        })

        if score > best_score:
            best_score = score
            best_cfg = dict(cfg)
            best_cfg["selected_indices"] = idx.tolist()

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pd.DataFrame(results).to_csv(
        output_dir / "hyperparameter_search.csv", index=False
    )

    if best_cfg is None:
        raise RuntimeError("Hyperparameter tuning failed.")

    return best_cfg


# ============================================================
# 6. Ablation and open-set evaluation
# ============================================================

def evaluate_open_set(
    model, val_loader, test_loader, unknown_loader, device
):
    # Threshold is calibrated only on known validation sequences.
    vy, vp, ve = predict(model, val_loader, device)
    known_val_energy = ve[vy >= 0]

    # Accept 95% of validation known examples, approximately.
    threshold = float(np.quantile(known_val_energy, 0.95))

    ty, tp, te = predict(model, test_loader, device)
    uy, up, ue = predict(model, unknown_loader, device)

    known_energy = te[ty >= 0]
    # Also treat test-split classes absent from the training vocabulary as unknown.
    unseen_test_energy = te[ty < 0]
    unknown_energy = np.concatenate([ue, unseen_test_energy])
    if len(known_energy) and len(unknown_energy):
        labels = np.concatenate([
            np.zeros(len(known_energy)),
            np.ones(len(unknown_energy)),
        ])
        scores = np.concatenate([known_energy, unknown_energy])

        try:
            open_auc = roc_auc_score(labels, scores)
        except ValueError:
            open_auc = np.nan

        unknown_recall = float(
            (unknown_energy > threshold).mean()
        )
        known_acceptance = float(
            (known_energy <= threshold).mean()
        )
    else:
        open_auc = np.nan
        unknown_recall = np.nan
        known_acceptance = np.nan

    return {
        **classification_metrics(ty, tp),
        "open_auroc_energy": open_auc,
        "unknown_recall_at_val95": unknown_recall,
        "known_acceptance_at_val95": known_acceptance,
        "energy_threshold": threshold,
        "n_known_test": int(len(known_energy)),
        "n_unknown_test": int(len(unknown_energy)),
        "n_unseen_label_test": int(len(unseen_test_energy)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--data",
    type=Path,
    default=Path(
        "/kaggle/input/datasets/kabil908/minimized"
    ),
)

    parser.add_argument(
        "--out", type=Path, default=Path("csrmt_results")
    )
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument(
        "--selection", choices=["fisher", "mi", "hybrid"],
        default="hybrid"
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--tune-trials", type=int, default=6)
    parser.add_argument("--tune-epochs", type=int, default=30)
    parser.add_argument("--tune-patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-tune", action="store_true")
    args, _ = parser.parse_known_args()

    if args.workers != 0:
        warnings.warn(
            "This streaming IterableDataset is designed for workers=0; "
            "forcing --workers 0 to avoid duplicated shard iteration."
        )
        args.workers = 0

    root = args.data
    # Strict minimized-only mode: do not silently fall back to any other dataset.
    if root.resolve() != Path("/kaggle/input/datasets/kabil908/minimized").resolve():
        raise ValueError(
            "CSRMT is configured for the minimized VeReMi dataset only. "
            "Use --data /kaggle/input/datasets/kabil908/minimized."
        )
    if not (root / "sequences").is_dir() or not list((root / "sequences").glob("train_*.npz")):
        raise FileNotFoundError(
            f"Minimized dataset shards not found under {root / 'sequences'}. "
            "Attach the minimized dataset to the Kaggle notebook and verify its path."
        )

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Data root:", root)
    print("Device:", device)

    train_files = get_split_files(root, "train")
    val_files = get_split_files(root, "validation")
    test_files = get_split_files(root, "test")
    unknown_files = get_split_files(root, "open_set_attack")

    label_map = load_label_vocabulary(train_files)
    seq_len, nf = infer_dimensions(train_files)
    nc = len(label_map)

    # Validate all shards have compatible tensor shapes before a long run.
    for split_name, split_files in [
        ("train", train_files), ("validation", val_files),
        ("test", test_files), ("open_set_attack", unknown_files),
    ]:
        for shard in split_files:
            with np.load(shard, allow_pickle=False) as z:
                if "X" not in z or "y_multiclass" not in z:
                    raise ValueError(f"{shard} must contain X and y_multiclass.")
                x_shape = z["X"].shape
                y_shape = z["y_multiclass"].shape
                if len(x_shape) != 3 or x_shape[1:] != (seq_len, nf):
                    raise ValueError(
                        f"Incompatible X shape in {shard}: {x_shape}; "
                        f"expected [N,{seq_len},{nf}]"
                    )
                if len(y_shape) != 1 or y_shape[0] != x_shape[0]:
                    raise ValueError(
                        f"Incompatible labels in {shard}: X={x_shape}, y={y_shape}"
                    )

    print(f"Sequence length={seq_len}, features={nf}, classes={nc}")

    # Normalize using training data only; apply identical statistics to all splits.
    feature_mean, feature_std = compute_train_feature_stats(train_files, nf)
    (out / "train_feature_normalization.json").write_text(
        json.dumps({
            "mean": feature_mean.tolist(),
            "std": feature_std.tolist(),
            "fit_split": "train_only",
        }, indent=2)
    )

    if nf != 20:
        raise ValueError(
            f"CSRMT minimized configuration expects 20 input features, but found {nf}. "
            "Check that the correct minimized VeReMi shards are attached."
        )

    # CSRMT consumes all 20 minimized VeReMi features. No feature ranking is
    # used, so the full-feature Transformer and relational pathway see the same input.
    ranked_features = np.arange(nf, dtype=np.int64)
    selected = ranked_features.copy()
    pd.DataFrame({"feature_index": ranked_features, "used_by_csrmt": True}).to_csv(
        out / "feature_ranking.csv", index=False
    )
    (out / "selected_features.json").write_text(json.dumps({
        "mode": "all_features", "selected_indices": selected.tolist(),
        "feature_count": nf, "top_k": nf,
    }, indent=2))
    print("CSRMT uses all feature indices:", selected.tolist())

    if args.no_tune:
        cfg = {
            "top_k": len(selected),
            "selected_indices": selected.tolist(),
            "lr": 5e-4,
            "weight_decay": 1e-4,
            "dropout": 0.2,
            "temperature": 0.1,
            "proto_weight": 0.1,
            "unknown_weight": 0.2,
        }
    else:
        cfg = tune_config(
            root, label_map, ranked_features, nf, nc, seq_len,
            args, device, out, feature_mean, feature_std,
        )
        # CSRMT always consumes all 20 features; tuning does not alter the input schema.
        selected = np.arange(nf, dtype=np.int64)
        cfg["selected_indices"] = selected.tolist()

    (out / "best_config.json").write_text(
        json.dumps(cfg, indent=2)
    )
    (out / "selected_features.json").write_text(
        json.dumps({
            "method": "all_features",
            "selected_indices": selected.tolist(),
            "feature_count": nf,
            "top_k": len(selected),
            "ranking_split": "not_applicable",
        }, indent=2)
    )
    print("Selected configuration:", cfg)

    train_loader = make_loader(
        train_files, label_map, args.batch_size, True, args.seed,
        feature_mean=feature_mean, feature_std=feature_std
    )
    val_loader = make_loader(
        val_files, label_map, args.batch_size, False, args.seed,
        feature_mean=feature_mean, feature_std=feature_std
    )
    test_loader = make_loader(
        test_files, label_map, args.batch_size, False, args.seed,
        feature_mean=feature_mean, feature_std=feature_std
    )
    unknown_loader = make_loader(
        unknown_files, label_map, args.batch_size, False, args.seed,
        feature_mean=feature_mean, feature_std=feature_std
    )

    variants = [
        "transformer_only",
        "no_temporal_cnn",
        "no_relation",
        "no_gate",
        "no_memory",
        "full_csrmt",
    ]
    all_results = []

    # Three seeds per ablation, with the same selected config.
    for variant in variants:
        for seed in SEEDS:
            print(f"\n{'=' * 70}")
            print(f"MODEL={variant} SEED={seed}")
            print(f"{'=' * 70}")

            seed_cfg = dict(cfg)
            indices = np.asarray(selected, dtype=np.int64)
            model = make_model(
                nf, indices, nc, seq_len, seed_cfg, variant
            )
            ckpt = out / f"{variant}_seed{seed}.pt"

            model, history, saved = train_model(
                model, train_loader, val_loader, device,
                seed_cfg, args.epochs, args.patience,
                ckpt, seed,
            )

            pd.DataFrame(history).to_csv(
                out / f"{variant}_seed{seed}_history.csv",
                index=False,
            )

            result = evaluate_open_set(
                model, val_loader, test_loader,
                unknown_loader, device,
            )
            result.update({
                "model": variant,
                "seed": seed,
                "best_epoch": saved["best_epoch"],
                "best_val_macro_f1": saved["best_val_macro_f1"],
                "parameters": sum(
                    p.numel() for p in model.parameters()
                ),
            })
            all_results.append(result)
            print(json.dumps(result, indent=2))

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    raw = pd.DataFrame(all_results)
    raw.to_csv(out / "all_seed_results.csv", index=False)

    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "mcc",
        "macro_ovr_auc",
        "open_auroc_energy",
        "unknown_recall_at_val95",
        "known_acceptance_at_val95",
        "best_val_macro_f1",
        "best_epoch",
        "parameters",
    ]

    summary_rows = []
    for model_name, group in raw.groupby("model"):
        row = {"model": model_name}
        for metric in metric_cols:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "summary_mean_std.csv", index=False)

    pretty_rows = []
    for _, row in summary.iterrows():
        result = {"model": row["model"]}
        for metric in metric_cols:
            mean = row[f"{metric}_mean"]
            std = row[f"{metric}_std"]
            result[metric] = (
                f"{mean:.4f} +/- {std:.4f}"
                if pd.notna(mean) and pd.notna(std)
                else "NA"
            )
        pretty_rows.append(result)

    pd.DataFrame(pretty_rows).to_csv(
        out / "summary_formatted.csv", index=False
    )

    (out / "experiment_contract.json").write_text(
        json.dumps({
            "seeds": SEEDS,
            "input_feature_mode": "all_20_minimized_features",
            "selected_features": selected.tolist(),
            "feature_selection": "not_used",
            "hyperparameter_tuning": not args.no_tune,
            "model_selection_metric": "validation_macro_f1",
            "early_stopping": True,
            "test_used_for_selection": False,
            "open_threshold_calibration": "validation_known_energy_95th_percentile",
            "ablation_models": variants,
            "architecture": "CSRMT",
            "temporal_cnn": {"kernels": [3, 5], "dilations": [1, 2, 4]},
            "relation_pathway": "learned feature-value and first/second temporal-difference attention; semantic column indices not hard-coded because no verified schema supplied",
            "prototype_memory": "class-conditional EMA momentum update",
            "unknown_head_training": "known train sequences vs synthetic perturbed/reversed train-only pseudo-unknowns",
            "transformer": {
                "d_model": 128,
                "heads": 4,
                "layers": 3,
                "ffn": 256,
            },
            "data_root": str(root.resolve()),
        }, indent=2)
    )

    print("\nFINAL MEAN +/- STD")
    print(summary.to_string(index=False))
    print("\nOutputs saved to:", out.resolve())


if __name__ == "__main__":
    main()