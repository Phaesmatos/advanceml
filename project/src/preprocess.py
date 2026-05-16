from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import numpy as np

from features import build_feature_selection, extract_yaml_feature_candidates
from utils import load_yaml, save_yaml


def build_raw_arrays(file_path: str, jet_features: List[str], track_features: List[str], max_jets: int | None = None):
    with h5py.File(file_path, "r") as f:
        jets_ds = f["jets"]
        tracks_ds = f["tracks"]
        n = jets_ds.shape[0] if max_jets is None else min(max_jets, jets_ds.shape[0])

        x_jets = np.stack([jets_ds[feat][:n] for feat in jet_features], axis=1).astype(np.float32)
        x_tracks = np.stack([tracks_ds[feat][:n] for feat in track_features], axis=2).astype(np.float32)
        track_mask = np.asarray(tracks_ds["valid"][:n]).astype(bool)

        metadata = {}
        for key in ["isDisplaced", "flavour_label", "mcEventWeight", "eventNumber"]:
            if key in jets_ds.dtype.names:
                metadata[key] = np.asarray(jets_ds[key][:n])
    return x_jets, x_tracks, track_mask, metadata


def clean_arrays(x_jets, x_tracks, track_mask, jet_features, track_features, clip_sigma: float | None = None):
    report = {"jets": {}, "tracks": {}}
    for i, feat in enumerate(jet_features):
        col = x_jets[:, i]
        report["jets"][feat] = {"nan": int(np.isnan(col).sum()), "inf": int(np.isinf(col).sum())}
        finite = np.isfinite(col)
        mean = float(np.mean(col[finite])) if finite.any() else 0.0
        col[~finite] = mean
        if clip_sigma is not None:
            std = float(np.std(col)) + 1e-8
            col[:] = np.clip(col, mean - clip_sigma * std, mean + clip_sigma * std)

    x_tracks = np.where(track_mask[..., None], x_tracks, 0.0)
    for i, feat in enumerate(track_features):
        col = x_tracks[:, :, i]
        valid_vals = col[track_mask]
        report["tracks"][feat] = {"nan": int(np.isnan(valid_vals).sum()), "inf": int(np.isinf(valid_vals).sum())}
        finite = np.isfinite(valid_vals)
        mean = float(np.mean(valid_vals[finite])) if finite.any() else 0.0
        bad = ~np.isfinite(col)
        col[bad] = mean
        if clip_sigma is not None:
            std = float(np.std(col[track_mask])) + 1e-8
            col[:] = np.clip(col, mean - clip_sigma * std, mean + clip_sigma * std)
    x_tracks = np.where(track_mask[..., None], x_tracks, 0.0)
    return x_jets, x_tracks, report


def normalize_features(X, feature_names, norm_dict: Dict, object_type: str, mask=None):
    stats = {}
    Xn = X.copy()
    for i, feat in enumerate(feature_names):
        entry = norm_dict.get(object_type, {}).get(feat, None)
        if mask is None:
            ref = Xn[:, i]
        else:
            ref = Xn[:, :, i][mask]
        if entry:
            mean, std = float(entry["mean"]), max(float(entry["std"]), 1e-8)
        else:
            mean, std = float(np.mean(ref)), max(float(np.std(ref)), 1e-8)
        if mask is None:
            Xn[:, i] = (Xn[:, i] - mean) / std
        else:
            Xn[:, :, i] = (Xn[:, :, i] - mean) / std
        stats[feat] = {"mean": mean, "std": std, "source": "norm_dict" if entry else "computed"}
    if mask is not None:
        Xn = np.where(mask[..., None], Xn, 0.0)
    return Xn.astype(np.float32), stats


def save_processed(out_path: Path, X_jets, X_tracks, track_mask, jet_feature_names, track_feature_names, metadata, y=None):
    payload = {
        "X_jets": X_jets,
        "X_tracks": X_tracks,
        "track_mask": track_mask,
        "jet_feature_names": np.array(jet_feature_names, dtype=object),
        "track_feature_names": np.array(track_feature_names, dtype=object),
    }
    for k, v in metadata.items():
        payload[f"meta_{k}"] = v
    if y is not None:
        payload["y"] = y
    np.savez_compressed(out_path, **payload)


def check_processed_dataset(processed_file: str) -> Dict[str, bool]:
    data = np.load(processed_file, allow_pickle=True)
    Xj, Xt, m = data["X_jets"], data["X_tracks"], data["track_mask"]
    checks = {
        "shapes_ok": Xj.ndim == 2 and Xt.ndim == 3 and m.shape == Xt.shape[:2] and m.shape[1] == 200,
        "jets_finite": np.isfinite(Xj).all(),
        "tracks_finite": np.isfinite(Xt).all(),
        "invalid_tracks_zero": np.allclose(Xt[~m], 0.0),
        "feature_names_ok": Xj.shape[1] == len(data["jet_feature_names"]) and Xt.shape[2] == len(data["track_feature_names"]),
    }
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="project/data/raw")
    parser.add_argument("--config-dir", default="project/configs")
    parser.add_argument("--out-dir", default="project/data/processed")
    parser.add_argument("--max-jets", type=int, default=None)
    args = parser.parse_args()

    files = [
        "pp_output_test_background.h5",
        "pp_output_test_signal.h5",
        "pp_output_val.h5",
    ]
    file_paths = [str(Path(args.raw_dir) / f) for f in files if (Path(args.raw_dir) / f).exists()]
    if not file_paths:
        raise FileNotFoundError("No HDF5 files found in raw-dir")

    yaml_candidates = extract_yaml_feature_candidates([
        str(Path(args.config_dir) / "ejs_train.yaml"),
        str(Path(args.config_dir) / "ejs_val.yaml"),
        str(Path(args.config_dir) / "ejs_test.yaml"),
    ])
    selection = build_feature_selection(file_paths, yaml_candidates)
    norm_dict = load_yaml(Path(args.config_dir) / "norm_dict.yaml")

    all_stats = {"nan_inf_report": {}, "normalization": {}}
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    for fp in file_paths:
        xj, xt, mask, metadata = build_raw_arrays(fp, selection.selected_jet_features, selection.selected_track_features, args.max_jets)
        xj, xt, report = clean_arrays(xj, xt, mask, selection.selected_jet_features, selection.selected_track_features)
        xj, jet_stats = normalize_features(xj, selection.selected_jet_features, norm_dict, "jets")
        xt, track_stats = normalize_features(xt, selection.selected_track_features, norm_dict, "tracks", mask=mask)

        y = None
        name = Path(fp).name
        if "background" in name:
            y = np.zeros(xj.shape[0], dtype=np.int64)
            out_name = "processed_background_test.npz"
        elif "signal" in name:
            y = np.ones(xj.shape[0], dtype=np.int64)
            out_name = "processed_signal_test.npz"
        else:
            out_name = "processed_val.npz"
            if "isDisplaced" in metadata:
                y = metadata["isDisplaced"].astype(np.int64)

        save_processed(Path(args.out_dir) / out_name, xj, xt, mask, selection.selected_jet_features, selection.selected_track_features, metadata, y)
        all_stats["nan_inf_report"][name] = report
        all_stats["normalization"][name] = {"jets": jet_stats, "tracks": track_stats}

    all_stats["selected_jet_features"] = selection.selected_jet_features
    all_stats["selected_track_features"] = selection.selected_track_features
    all_stats["metadata_fields"] = selection.metadata_fields
    all_stats["label_fields"] = selection.label_fields
    all_stats["excluded_fields"] = selection.excluded_fields
    save_yaml(all_stats, Path(args.out_dir) / "preprocessing_stats.yaml")


if __name__ == "__main__":
    main()
