"""Shared query-type router for ScanRefer and Nr3D experiments.

The router is intentionally dataset-agnostic: ScanRefer rows usually expose
`query_type`, while Nr3D parser rows expose `relation_source`. Both are mapped
onto the same evidence type and then to the same input representation.
"""

from __future__ import annotations

import re


E0_ROUTE = "E0"

ROUTE_POLICY = {
    "explicit_direction": E0_ROUTE,
    "proximity_derived": "seeground_ablation_spatial_only",
    "none": E0_ROUTE,
    "ordinal": "bev_raw_labeled",
    "geometric": "seeground_ablation_3dpos_only",
    "object_orientation": E0_ROUTE,
    "room_side": E0_ROUTE,
    "uncategorized": E0_ROUTE,
    "opposite_derived": E0_ROUTE,
    "viewpoint_guided": "bev_raw_labeled",
}

VISUAL_ATTRIBUTE_TERMS = {
    "red", "blue", "green", "yellow", "orange", "white", "black", "brown",
    "gray", "grey", "purple", "pink", "color", "colored", "colour",
    "dark", "light", "bright", "wooden", "wood", "metal", "plastic",
    "glass", "fabric", "leather", "open", "closed", "seat up", "is up",
    "on top", "on it", "bag", "note", "towel", "objects on top",
    "most objects",
}

SHAPE_ATTRIBUTE_TERMS = {
    "round", "square", "rectangular", "oval",
}

PURE_ORDINAL_PATTERNS = (
    r"\bmiddle\b",
    r"\bcenter\b",
    r"\bcentre\b",
    r"\bleftmost\b",
    r"\brightmost\b",
    r"\bsecond\b",
    r"\bthird\b",
    r"\bfirst\b",
    r"\blast\b",
    r"\bin the corner\b",
    r"\bright stall\b",
    r"\bleft stall\b",
    r"\bsmallest\b",
    r"\bbiggest\b",
    r"\blargest\b",
    r"\btallest\b",
    r"\bshortest\b",
    r"\bshorter\b",
    r"\btaller\b",
)

GEOMETRIC_PATTERNS = (
    r"\bbetween\b",
    r"\bsurrounded by\b",
    r"\bin between\b",
    r"\babove\b",
    r"\bunder\b",
    r"\bbelow\b",
    r"\binside\b",
)


def normalize_query_type(value: str | None) -> str:
    if not value or value == "NA":
        return "uncategorized"
    return str(value)


def normalize_text(value: str | None) -> str:
    return (value or "").lower().replace("-", " ")


def query_text_from_row(row: dict) -> str:
    return str(row.get("query") or row.get("caption") or row.get("description") or "")


def has_visual_attribute(text: str | None, *, include_shape: bool = True) -> bool:
    normalized = normalize_text(text)
    terms = set(VISUAL_ATTRIBUTE_TERMS)
    if include_shape:
        terms.update(SHAPE_ATTRIBUTE_TERMS)
    return any(term in normalized for term in terms)


def matches_any(text: str | None, patterns: tuple[str, ...]) -> bool:
    normalized = normalize_text(text)
    return any(re.search(pattern, normalized) for pattern in patterns)


def is_pure_ordinal_query(text: str | None) -> bool:
    return not has_visual_attribute(text) and matches_any(text, PURE_ORDINAL_PATTERNS)


def is_pure_geometric_query(text: str | None) -> bool:
    return not has_visual_attribute(text) and matches_any(text, GEOMETRIC_PATTERNS)


def query_type_from_row(row: dict) -> str:
    """Read the shared query type from either ScanRefer or Nr3D metadata."""
    return normalize_query_type(
        row.get("query_type")
        or row.get("relation_source")
        or row.get("source")
    )


def route_for_query_type(query_type: str | None) -> str:
    return ROUTE_POLICY.get(normalize_query_type(query_type), E0_ROUTE)


def route_for_row(row: dict) -> tuple[str, str]:
    query_type = query_type_from_row(row)
    return query_type, route_for_query_type(query_type)


def route_for_row_universal(row: dict) -> tuple[str, str, str]:
    """Dataset-agnostic evidence-aware policy.

    This policy intentionally avoids dataset-specific calibration. It routes by
    the evidence required by the query text and uses E0 as the default whenever
    visual attributes, mixed evidence, or unresolved viewpoint frames are likely.
    """
    query_type = query_type_from_row(row)
    text = query_text_from_row(row)

    if query_type == "proximity_derived":
        return query_type, "seeground_ablation_spatial_only", "pure_proximity_spatial"
    if has_visual_attribute(text):
        return query_type, E0_ROUTE, "visual_attribute_default_e0"
    if query_type == "ordinal" and is_pure_ordinal_query(text):
        return query_type, "bev_raw_labeled", "pure_ordinal_bev"
    if query_type == "geometric" and is_pure_geometric_query(text):
        return query_type, "seeground_ablation_3dpos_only", "pure_geometric_3dpos"
    if query_type == "viewpoint_guided":
        return query_type, E0_ROUTE, "viewpoint_needs_local_frame_default_e0"
    return query_type, E0_ROUTE, "default_e0"
