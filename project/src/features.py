from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Set

from inspect_h5 import get_dataset_fields
from utils import flatten_feature_values, load_yaml

EXCLUDED_INPUT_FIELDS = {
    "isDisplaced",
    "flavour_label",
    "isTagged",
    "salt_pdisp",
    "mcEventWeight",
    "eventNumber",
    "truthOriginLabel",
    "truthVertexIndex",
    "VSIVertexIndex",
    "truth_dark_pions",
    "truth_stable_non_geant",
    "valid",
}

METADATA_FIELDS = ["isDisplaced", "flavour_label", "mcEventWeight", "eventNumber"]
LABEL_FIELDS = ["isDisplaced"]


@dataclass
class FeatureSelectionResult:
    selected_jet_features: List[str]
    selected_track_features: List[str]
    metadata_fields: List[str]
    label_fields: List[str]
    excluded_fields: Dict[str, str]


def extract_yaml_feature_candidates(config_paths: Iterable[str]) -> Set[str]:
    candidates: Set[str] = set()
    for path in config_paths:
        data = load_yaml(path)
        for key in ["features", "jet_features", "track_features", "inputs", "variables"]:
            if key in data:
                candidates.update(flatten_feature_values(data[key]))
        candidates.update(flatten_feature_values(data))
    return {c for c in candidates if isinstance(c, str)}


def build_feature_selection(
    files: Iterable[str],
    yaml_feature_candidates: Set[str],
) -> FeatureSelectionResult:
    jet_sets = [set(get_dataset_fields(fp, "jets")) for fp in files]
    track_sets = [set(get_dataset_fields(fp, "tracks")) for fp in files]
    common_jets = set.intersection(*jet_sets)
    common_tracks = set.intersection(*track_sets)

    excluded = {f: "excluded_by_rule" for f in EXCLUDED_INPUT_FIELDS if f in common_jets or f in common_tracks}

    jet_features = sorted([
        f for f in common_jets
        if f not in EXCLUDED_INPUT_FIELDS and (f in yaml_feature_candidates or not yaml_feature_candidates)
    ])
    track_features = sorted([
        f for f in common_tracks
        if f not in EXCLUDED_INPUT_FIELDS and (f in yaml_feature_candidates or not yaml_feature_candidates)
    ])

    return FeatureSelectionResult(
        selected_jet_features=jet_features,
        selected_track_features=track_features,
        metadata_fields=METADATA_FIELDS,
        label_fields=LABEL_FIELDS,
        excluded_fields=excluded,
    )
