from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from hqml_topological_qaoa import audit, experiment


class AuditContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.code_root = Path(__file__).resolve().parents[1]
        cls.data_root = cls.code_root / "data" / "source"
        cls.result = audit.run_audit(cls.data_root)

    def test_locked_results_reproduce(self) -> None:
        expected = json.loads(
            (self.code_root / "results" / "results.json").read_text(encoding="utf-8")
        )
        audit._assert_close(
            audit._without_timing(self.result), audit._without_timing(expected)
        )

    def test_all_eight_source_hashes(self) -> None:
        self.assertEqual(set(self.result["sources"]), set(audit.ALL_SOURCES))
        for name, metadata in audit.DATASETS.items():
            self.assertEqual(self.result["sources"][name]["sha256"], metadata["sha256"])

    def test_duplicate_topologies_are_excluded(self) -> None:
        self.assertEqual(
            self.result["excluded_topology_duplicates"], {"G6": "G1", "G18": "G14"}
        )
        self.assertEqual(self.result["records"], 144)

    def test_collision_free_grouped_protocol(self) -> None:
        grouped = self.result["protocols"]["signature_grouped"]
        self.assertEqual(grouped["feature_signature_overlap_total"], 0)
        self.assertEqual(grouped["zero_distance_test_records_total"], 0)

    def test_legacy_row_split_is_collision_dominated(self) -> None:
        legacy = self.result["legacy_row_split"]
        self.assertEqual(legacy["feature_collision_test_records"], 60)
        self.assertEqual(legacy["test_records"], 64)
        self.assertEqual(legacy["raw_neighborhood_duplicate_test_records"], 18)

    def test_theorem_checks(self) -> None:
        self.assertTrue(all(self.result["theorem_checks"].values()))

    def test_hierarchy_matches_dense_at_rank_eight(self) -> None:
        row = next(
            item for item in self.result["operator_audit"]["rank_audit"]
            if item["rank"] == 8
        )
        self.assertLess(row["relative_frobenius_error"], 1e-5)
        self.assertEqual(row["prediction_bound_violations"], 0)
        self.assertLess(row["storage_fraction"], 0.30)

    def test_depth_one_formula_matches_statevector(self) -> None:
        edges = [(0, 1), (1, 2), (0, 2)]
        diagonal = experiment.cost_diagonal(edges, 3)
        for gamma, beta in ((0.2, 0.1), (0.8, 0.4), (1.1, 0.25)):
            analytic = experiment.p1_closed_form(edges, gamma, beta, 3)
            numerical = experiment.schedule_energy(
                diagonal, np.array([gamma]), np.array([beta]), 3
            )
            self.assertAlmostEqual(analytic, numerical, places=12)

    def test_label_scope_is_explicit(self) -> None:
        contract = self.result["label_contract"]
        self.assertFalse(contract["weights_used"])
        self.assertFalse(contract["p2_is_global_optimum"])
        self.assertEqual(contract["p2_local_refinement_queries"], 25)


if __name__ == "__main__":
    unittest.main()
