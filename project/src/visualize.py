from __future__ import annotations

import h5py
import matplotlib.pyplot as plt
import numpy as np


def _read_fields(file_path: str, dataset: str, features: list[str], max_jets: int | None = None):
    with h5py.File(file_path, "r") as f:
        ds = f[dataset]
        n = ds.shape[0] if max_jets is None else min(max_jets, ds.shape[0])
        return {feat: np.asarray(ds[feat][:n]) for feat in features}


def plot_jet_distributions(file_path: str, features: list[str], max_jets: int = 100000, bins: int = 80):
    data = _read_fields(file_path, "jets", features, max_jets=max_jets)
    ncols = 3
    nrows = int(np.ceil(len(features) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axs = np.atleast_1d(axs).reshape(-1)
    for i, feat in enumerate(features):
        x = data[feat].reshape(-1)
        x = x[np.isfinite(x)]
        axs[i].hist(x, bins=bins, histtype="step", linewidth=1.5)
        axs[i].set_title(feat)
    for j in range(i + 1, len(axs)):
        axs[j].axis("off")
    plt.tight_layout()


def plot_track_feature_distribution(file_path: str, feature: str, max_jets: int = 20000, bins: int = 100):
    with h5py.File(file_path, "r") as f:
        tracks = f["tracks"]
        n = min(max_jets, tracks.shape[0])
        vals = np.asarray(tracks[feature][:n])
        valid = np.asarray(tracks["valid"][:n]).astype(bool)
        x = vals[valid]
    x = x[np.isfinite(x)]
    plt.figure(figsize=(6, 4))
    plt.hist(x, bins=bins, histtype="step", linewidth=1.5)
    plt.title(feature)
    plt.tight_layout()


def plot_track_scatter_for_jet(file_path: str, jet_idx: int, color_feature: str = "d0"):
    with h5py.File(file_path, "r") as f:
        t = f["tracks"][jet_idx]
        valid = t["valid"].astype(bool)
        x = t["deta"][valid]
        y = t["dphi"][valid]
        c = t[color_feature][valid]
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(x, y, c=c, s=20, cmap="viridis", alpha=0.9)
    plt.colorbar(sc, label=color_feature)
    plt.xlabel("deta")
    plt.ylabel("dphi")
    plt.tight_layout()


def plot_track_feature_heatmap(file_path: str, jet_idx: int, features: list[str]):
    with h5py.File(file_path, "r") as f:
        t = f["tracks"][jet_idx]
        valid = t["valid"].astype(bool)
        matrix = np.stack([t[feat] for feat in features], axis=0)[:, valid]
    plt.figure(figsize=(10, 5))
    plt.imshow(matrix, aspect="auto", interpolation="nearest")
    plt.yticks(range(len(features)), features)
    plt.xlabel("valid track index")
    plt.colorbar()
    plt.tight_layout()
