from __future__ import annotations

COMPONENTS = (
    "target_availability",
    "target_exposure",
    "output_validity",
    "grounding_outcome",
)
PROVENANCE = {"Native", "Reconstructed", "Sensitivity-only", "Unobserved", "N/A"}
STATES = {
    "target_unavailable",
    "target_unexposed",
    "no_valid_output",
    "incorrect_grounding_outcome",
    "grounding_success",
    "unobserved",
}


class ContractError(ValueError):
    pass


def validate(row):
    required = ("schema_version", "query_id", "system", "dataset", "components", "provenance")
    missing = [key for key in required if key not in row]
    if missing:
        raise ContractError(f"missing required fields: {missing}")
    if row["schema_version"] != "1.0":
        raise ContractError(f"unsupported schema_version={row['schema_version']!r}")
    if not str(row["query_id"]):
        raise ContractError("query_id must be non-empty")
    for component in COMPONENTS:
        value = row["components"].get(component)
        if value not in (True, False, None):
            raise ContractError(f"{component} must be true, false, or null")
        provenance = row["provenance"].get(component, "Unobserved")
        if provenance not in PROVENANCE:
            raise ContractError(f"invalid provenance for {component}: {provenance}")
        if value is None and provenance not in {"Unobserved", "N/A"}:
            raise ContractError(f"null {component} requires Unobserved or N/A provenance")
    endpoint = row.get("endpoint_success")
    if endpoint not in (True, False, None):
        raise ContractError("endpoint_success must be true, false, or null")
    return row


def terminal_state(row):
    c = row["components"]
    if c["target_availability"] is False:
        return "target_unavailable"
    if c["target_availability"] is None:
        return "unobserved"
    if c["target_exposure"] is False:
        return "target_unexposed"
    if c["target_exposure"] is None:
        return "unobserved"
    if c["output_validity"] is False:
        return "no_valid_output"
    if c["output_validity"] is None:
        return "unobserved"
    if c["grounding_outcome"] is False:
        return "incorrect_grounding_outcome"
    if c["grounding_outcome"] is True:
        return "grounding_success"
    return "unobserved"

