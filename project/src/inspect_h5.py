from __future__ import annotations

from typing import Dict, Iterable, List

import h5py
import numpy as np


def inspect_h5_structure(file_path: str) -> None:
    def _visitor(name: str, obj) -> None:
        if isinstance(obj, h5py.Dataset):
            print(f"[DATASET] {name}: shape={obj.shape}, dtype={obj.dtype}")
            if obj.dtype.names:
                print(f"  fields={obj.dtype.names}")
        else:
            print(f"[GROUP]   {name}")

    print(f"\n=== HDF5 STRUCTURE: {file_path} ===")
    with h5py.File(file_path, "r") as f:
        f.visititems(_visitor)


def get_dataset_fields(file_path: str, dataset_name: str) -> List[str]:
    with h5py.File(file_path, "r") as f:
        ds = f[dataset_name]
        if ds.dtype.names is None:
            raise ValueError(f"Dataset '{dataset_name}' has no structured fields")
        return list(ds.dtype.names)


def compare_fields(files: Iterable[str], dataset_name: str) -> Dict[str, List[str]]:
    by_file: Dict[str, set[str]] = {
        fp: set(get_dataset_fields(fp, dataset_name)) for fp in files
    }
    all_fields = set.union(*by_file.values()) if by_file else set()
    common = set.intersection(*by_file.values()) if by_file else set()

    only_per_file = {
        fp: sorted(fields - set.union(*[v for k, v in by_file.items() if k != fp]))
        for fp, fields in by_file.items()
    }
    only_test = sorted(by_file.get("pp_output_test_background.h5", set()) - by_file.get("pp_output_val.h5", set()))

    out = {
        "common": sorted(common),
        "all": sorted(all_fields),
        "only_per_file": only_per_file,
        "only_test_background_vs_val": only_test,
    }
    return out


def summarize_feature(
    file_path: str,
    dataset_name: str,
    feature: str,
    max_entries: int = 100_000,
) -> Dict[str, float]:
    with h5py.File(file_path, "r") as f:
        arr = f[dataset_name][feature]
        flat = np.asarray(arr).reshape(-1)
    if max_entries and flat.size > max_entries:
        idx = np.linspace(0, flat.size - 1, max_entries, dtype=int)
        flat = flat[idx]

    finite = np.isfinite(flat)
    finite_vals = flat[finite]
    return {
        "n_total": int(flat.size),
        "n_nan": int(np.isnan(flat).sum()),
        "n_inf": int(np.isinf(flat).sum()),
        "min": float(np.min(finite_vals)) if finite_vals.size else float("nan"),
        "max": float(np.max(finite_vals)) if finite_vals.size else float("nan"),
        "mean": float(np.mean(finite_vals)) if finite_vals.size else float("nan"),
        "std": float(np.std(finite_vals)) if finite_vals.size else float("nan"),
    }
