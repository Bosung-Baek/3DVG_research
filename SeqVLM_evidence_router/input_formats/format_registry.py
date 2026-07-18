"""Final input-format registry for the standalone evidence-router repo."""

from .baseline_e0 import BaselineE0Format
from .bev_raw_labeled import BevRawLabeledFormat
from .seeground_ablation_spatial_only import SeeGroundAblationSpatialOnlyFormat


INPUT_FORMATS = {
    "baseline_e0": BaselineE0Format,
    "E0": BaselineE0Format,
    "bev_raw_labeled": BevRawLabeledFormat,
    "seeground_ablation_3dpos_only": SeeGroundAblationSpatialOnlyFormat,
    "seeground_ablation_spatial_only": SeeGroundAblationSpatialOnlyFormat,
}


def get_format(name: str):
    if name not in INPUT_FORMATS:
        raise ValueError(f"Unknown format: {name!r}. Available: {sorted(INPUT_FORMATS)}")
    return INPUT_FORMATS[name]()
