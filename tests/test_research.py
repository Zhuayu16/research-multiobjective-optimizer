from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from mtpv_optimizer.core import ideal_distances, load_table, validate_dataset
from mtpv_optimizer.coupled import FuelDefinition, OBJECTIVES, fit_coupled
from mtpv_optimizer.domain import SampleDomain
from mtpv_optimizer.nsga2 import _non_dominated_sort, _repair, optimize_nsga2
from mtpv_optimizer.workflow import RunConfig, export_workbook, run_workflow


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless((ROOT / "data" / "research_46_20261002.json").is_file(), "requires the private 46-case research dataset")
class ResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = load_table(ROOT / "data" / "research_46_20261002.json")
        cls.variables = ["L_mm", "phi", "qm_1e5_kg_s"]
        cls.bundle = fit_coupled(cls.frame, cls.variables, FuelDefinition(1, 2), model_mode="GaussianProcess")

    def test_mass_flow_units_are_equivalent(self):
        x = self.frame[self.variables].to_numpy(float)
        scaled = x.copy()
        scaled[:, 2] *= 1e-5
        np.testing.assert_allclose(FuelDefinition(1, 2).power(x), FuelDefinition(1, 2, 1).power(scaled))
        # Exported inlet fractions include rounded composition values.
        np.testing.assert_allclose(FuelDefinition(1, 2).fraction(x), self.frame["Y_H2_in"], rtol=1e-3)

    def test_coupled_identity_and_oof_errors(self):
        x = self.frame[self.variables].to_numpy(float)
        raw = self.bundle.predict_components(x)
        output = self.bundle.predict(x)
        np.testing.assert_allclose(output[:, 1], raw[:, 1] + raw[:, 2])
        np.testing.assert_allclose(output[:, 2], 100 * output[:, 1] / self.bundle.fuel.power(x))
        validation = self.bundle.validation_frame
        np.testing.assert_allclose(validation.loc[validation["目标"] == OBJECTIVES[1], "相对误差(%)"],
                                   validation.loc[validation["目标"] == OBJECTIVES[2], "相对误差(%)"])
        self.assertEqual(set(validation["fold"]), {1, 2, 3, 4, 5})
        self.assertAlmostEqual(self.bundle.metrics[0].mape_cv_pct, 0.541647751, places=4)
        self.assertAlmostEqual(self.bundle.metrics[1].mape_cv_pct, 1.09432155, places=4)

    def test_bad_fuel_definition_rejected(self):
        with self.assertRaisesRegex(ValueError, "Y_H2_in"):
            fit_coupled(self.frame, self.variables, FuelDefinition(1, 2, stoichiometric_air_fuel_ratio=30))
        with self.assertRaisesRegex(ValueError, "eta_gross_pct"):
            fit_coupled(self.frame.drop(columns="Y_H2_in"), self.variables, FuelDefinition(1, 2, 1))

    def test_no_silent_clipping_or_hull_extrapolation(self):
        x = self.frame[self.variables].to_numpy(float)
        outside = x.min(axis=0) - 0.1
        self.assertTrue(np.isnan(self.bundle.predict(outside)).all())
        domain = SampleDomain([[0, 0], [1, 0], [0, 1]], [0, 0], [1, 1])
        self.assertFalse(domain.contains([0.9, 0.9])[0])
        np.testing.assert_allclose(domain.repair(np.array([0.9, 0.9]), np.array([0.1, 0.1])).sum(), 1, atol=1e-8)
        samples = domain.sample(np.random.default_rng(3), 100)
        self.assertTrue(domain.contains(samples).all())
        with self.assertRaisesRegex(ValueError, "rank deficient"):
            SampleDomain([[0, 0], [1, 1], [2, 2]], [0, 0], [2, 2])

    def test_fixed_anchors_and_weights(self):
        low = np.array([227.020954, 6.3122919, 5.92596128])
        high = np.array([663.392046, 12.8958395, 8.10363089])
        actual = self.frame[OBJECTIVES].to_numpy(float)
        fitness = actual * [1, -1, -1]
        front = _non_dominated_sort(fitness)[0][0]
        for weights, expected in [((1, 1, 1), "OPT01"), ((2, 1, 1), "OPT04"),
                                  ((1, 2, 1), "T04"), ((1, 1, 2), "OPT01")]:
            distances = ideal_distances(actual, weights, (low, high))
            chosen = front[np.argmin(distances[front])]
            self.assertEqual(self.frame.iloc[chosen]["case"], expected)
            self.assertAlmostEqual(distances[chosen], ideal_distances(actual[[chosen]], weights, (low, high))[0])
        for bad in ((0, 0, 0), (np.nan, 1, 1), (-1, 1, 1), (1, 2)):
            with self.assertRaises(ValueError):
                ideal_distances(actual, bad, (low, high))

    def test_search_feasible_and_deterministic(self):
        anchors = (self.frame[OBJECTIVES].min().to_numpy(), self.frame[OBJECTIVES].max().to_numpy())
        a = optimize_nsga2(self.bundle, 20, 10, 17, runs=2, decision_method="IDEAL", anchors=anchors)
        b = optimize_nsga2(self.bundle, 20, 10, 17, runs=2, decision_method="IDEAL", anchors=anchors)
        np.testing.assert_allclose(a.variables, b.variables)
        self.assertTrue(self.bundle.domain.contains(a.variables).all())
        self.assertEqual([row["seed"] for row in a.run_summary], [17, 7936])
        self.assertEqual(len(_non_dominated_sort(a.objectives * [1, -1, -1])[0][0]), len(a.objectives))

    def test_repair_integer_single_and_multiple(self):
        np.testing.assert_equal(_repair(np.array([2.8, 3.7]), np.array([2.1, 3.1]), np.array([3.8, 4.8]), (0, 1)), [3, 4])

    def test_conflicting_inputs_rejected(self):
        duplicate = self.frame.iloc[[0]].copy()
        duplicate["dp_Pa"] *= 1.1
        with self.assertRaisesRegex(ValueError, "Duplicate input"):
            validate_dataset(pd.concat([self.frame, duplicate]), self.variables, ["dp_Pa", "P_MTPV_W", "P_MTEG_W"])

    def test_workflow_export_preserves_definitions(self):
        config = RunConfig(self.variables, ["dp_Pa"], self.frame[self.variables].min().tolist(),
                           self.frame[self.variables].max().tolist(), mode="coupled", model_mode="GaussianProcess",
                           population_size=20, generations=10, runs=1, check_endpoints=True, endpoint_iterations=3,
                           holdout_batch="3.3_confirmation")
        output = run_workflow(self.frame, config)
        self.assertEqual(set(output.observed["结果类型"]), {"输入表计算值"})
        self.assertEqual(set(output.predicted["结果类型"]), {"代理模型预测"})
        self.assertEqual(len(output.bundle.holdout_frame), 12)
        self.assertEqual(set(output.bundle.holdout_frame["训练组数"]), {42})
        self.assertEqual(len(output.endpoints), 4)
        self.assertTrue(output.bundle.domain.contains(output.predicted[self.variables].to_numpy()).all())
        with TemporaryDirectory() as directory:
            path = Path(directory) / "result.xlsx"
            export_workbook(output, path)
            saved = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(saved["normalized_weights"], [1 / 3] * 3)
            self.assertEqual(saved["sample_count"], 46)
            self.assertEqual(len(saved["clean_data_sha256"]), 64)
            with pd.ExcelFile(path) as excel:
                sheets = excel.sheet_names
            self.assertIn("已计算Pareto", sheets)
            self.assertIn("模型预测Pareto", sheets)
            self.assertIn("响应留出检验", sheets)


if __name__ == "__main__":
    unittest.main()
