import json
import tempfile
import unittest
from pathlib import Path

from epp_toolkit.adapters import canonical
from epp_toolkit.audit import aggregate, write_report
from epp_toolkit.contract import ContractError, terminal_state


def row(qid, ta, te, ov, go, endpoint):
    return canonical({
        "schema_version": "1.0", "query_id": qid, "system": "fixture", "dataset": "fixture",
        "components": {"target_availability": ta, "target_exposure": te, "output_validity": ov, "grounding_outcome": go},
        "endpoint_success": endpoint, "execution_status": "completed",
        "provenance": {k: "Native" if v is not None else "Unobserved" for k, v in {
            "target_availability": ta, "target_exposure": te, "output_validity": ov, "grounding_outcome": go}.items()},
    })


class CoreTest(unittest.TestCase):
    def test_states_and_denominators(self):
        rows = [row("1", False, False, True, False, False), row("2", True, False, True, False, False),
                row("3", True, True, False, False, False), row("4", True, True, True, False, False),
                row("5", True, True, True, True, True)]
        self.assertEqual([terminal_state(x) for x in rows], ["target_unavailable", "target_unexposed", "no_valid_output", "incorrect_grounding_outcome", "grounding_success"])
        p = aggregate(rows)["metrics"]
        self.assertEqual((p["target_availability_rate"]["numerator"], p["target_availability_rate"]["denominator"]), (4, 5))
        self.assertEqual((p["target_exposure_rate"]["numerator"], p["target_exposure_rate"]["denominator"]), (3, 4))
        self.assertEqual((p["conditional_grounding_accuracy"]["numerator"], p["conditional_grounding_accuracy"]["denominator"]), (1, 2))

    def test_unobserved_is_not_failure(self):
        r = row("1", None, None, True, None, False)
        self.assertEqual(terminal_state(r), "unobserved")
        p = aggregate([r])["metrics"]
        self.assertEqual(p["target_availability_rate"]["denominator"], 0)

    def test_report_is_deterministic(self):
        rows = [row("1", True, True, True, True, True)]
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / "source.jsonl"; source.write_text(json.dumps(rows[0]) + "\n")
            a, b = Path(d) / "a", Path(d) / "b"
            write_report(rows, [source], a); write_report(rows, [source], b)
            for name in ("profile.json", "records.jsonl", "observability.json", "profile.csv"):
                self.assertEqual((a/name).read_bytes(), (b/name).read_bytes())


if __name__ == "__main__":
    unittest.main()
