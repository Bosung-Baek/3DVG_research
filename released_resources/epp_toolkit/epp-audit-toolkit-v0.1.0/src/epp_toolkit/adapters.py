from __future__ import annotations

from .contract import COMPONENTS, validate


def base(query_id, system, dataset, scene_id, components, endpoint, provenance, execution="completed", metadata=None):
    row = {
        "schema_version": "1.0",
        "query_id": str(query_id),
        "system": system,
        "dataset": dataset,
        "scene_id": scene_id,
        "components": {name: components.get(name) for name in COMPONENTS},
        "endpoint_success": endpoint,
        "execution_status": execution,
        "provenance": {name: provenance.get(name, "Unobserved") for name in COMPONENTS},
        "metadata": metadata or {},
    }
    return validate(row)


def canonical(row):
    return validate(row)


def seqvlm_exp06(row):
    prov = {name: "Native" for name in COMPONENTS}
    return base(
        row.get("row_id") or row.get("query_id"), "SeqVLM-E0", row.get("dataset"), row.get("scene_id"),
        {"target_availability": bool(row.get("ta")), "target_exposure": bool(row.get("te")),
         "output_validity": bool(row.get("ov")), "grounding_outcome": bool(row.get("da"))},
        bool(row.get("e2e")), prov,
        metadata={"condition": row.get("condition"), "query_type": row.get("query_type"),
                  "native_candidate_count": len(row.get("native_candidate_ids") or []),
                  "actual_input_candidate_count": len(row.get("actual_input_candidate_ids") or [])},
    )


def seqvlm_profile(row):
    prov = {name: "Native" for name in COMPONENTS}
    return base(
        f"{row.get('dataset')}:{row.get('policy')}:{row.get('query_id')}",
        f"SeqVLM-{row.get('policy', 'unknown')}", row.get("dataset"), row.get("scene_id"),
        {"target_availability": row.get("available"), "target_exposure": row.get("exposed"),
         "output_validity": row.get("pipeline_success"), "grounding_outcome": row.get("success")},
        row.get("success"), prov, metadata={"query_type": row.get("query_type"), "route": row.get("selected_route")},
    )


def csvg_compatible(row):
    components = {
        "target_availability": row.get("availability"),
        "target_exposure": row.get("exposure") if row.get("exposure_evaluable") else None,
        "output_validity": row.get("output_valid"),
        "grounding_outcome": row.get("decision_success") if row.get("decision_evaluable") else None,
    }
    prov = {name: ("Sensitivity-only" if value is not None else "Unobserved")
            for name, value in components.items()}
    return base(
        f"csvg:{row.get('row_index')}", "CSVG-compatible", "scanrefer", row.get("scene_id"),
        components,
        row.get("endpoint_success"), prov, metadata={"replay_status": row.get("replay_status")},
    )


ADAPTERS = {
    "canonical": canonical,
    "seqvlm-exp06": seqvlm_exp06,
    "seqvlm-profile": seqvlm_profile,
    "csvg-compatible": csvg_compatible,
}
