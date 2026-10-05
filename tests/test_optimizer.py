from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from mtpv_optimizer.core import fit_surrogates, validate_dataset
from mtpv_optimizer.dat_parser import import_dat_files
from mtpv_optimizer.doe import central_composite_design
from mtpv_optimizer.nsga2 import optimize_nsga2


class OptimizerTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        count = 70
        phi = rng.uniform(0.8, 1.2, count)
        flow = rng.uniform(3.5, 5.5, count)
        length = rng.uniform(5.0, 25.0, count)
        pressure = 100 + 40 * flow**2 + 2.5 * length - 35 * phi
        power = 1.0 + 0.7 * flow + 0.08 * length + 1.5 * phi
        efficiency = 4.5 - 0.12 * (flow - 4.2) ** 2 - 0.002 * (length - 14) ** 2 + 0.4 * phi
        self.frame = pd.DataFrame(
            {
                "当量比": phi,
                "入口质量流量": flow,
                "翅片长度L": length,
                "压降": pressure,
                "能量输出": power,
                "系统效率": efficiency,
            }
        )

    def test_ccd_three_variables(self) -> None:
        design = central_composite_design(
            ["当量比", "入口质量流量", "翅片长度L"],
            np.array([0.8, 3.5, 5.0]),
            np.array([1.2, 5.5, 25.0]),
        )
        self.assertEqual(len(design), 25)
        self.assertTrue((design["当量比"].between(0.8, 1.2)).all())

    def test_constant_variable_is_rejected(self) -> None:
        frame = self.frame.copy()
        frame["入口质量流量"] = 4.5
        with self.assertRaisesRegex(ValueError, "4.5"):
            validate_dataset(
                frame,
                ["当量比", "入口质量流量", "翅片长度L"],
                ["压降", "能量输出", "系统效率"],
            )

    def test_fit_and_optimize(self) -> None:
        variables = ["当量比", "入口质量流量", "翅片长度L"]
        targets = ["压降", "能量输出", "系统效率"]
        clean, _ = validate_dataset(self.frame, variables, targets)
        bundle = fit_surrogates(clean, variables, targets, seed=3)
        self.assertTrue(all(metric.r2_cv > 0.99 for metric in bundle.metrics))
        self.assertTrue({"MLP-NeuralNet", "RSM-Quadratic", "GaussianProcess"}.issubset(set(bundle.candidate_metrics["模型"])))
        result = optimize_nsga2(bundle, population_size=40, generations=15, seed=3, runs=2)
        self.assertGreater(len(result.variables), 1)
        self.assertEqual(result.objectives.shape[1], 3)
        self.assertTrue(np.all(result.variables >= bundle.lower - 1e-12))
        self.assertTrue(np.all(result.variables <= bundle.upper + 1e-12))
        self.assertEqual(result.recommended_index, 0)

    def test_explicit_neural_network_model(self) -> None:
        variables = ["当量比", "入口质量流量", "翅片长度L"]
        targets = ["压降", "能量输出", "系统效率"]
        clean, _ = validate_dataset(self.frame, variables, targets)
        bundle = fit_surrogates(clean, variables, targets, seed=4, model_mode="MLP-NeuralNet")
        self.assertTrue(all(metric.model_name == "MLP-NeuralNet" for metric in bundle.metrics))
        self.assertEqual(len(bundle.validation_frame), len(clean) * 3)

    def test_dat_profile_import(self) -> None:
        content = """[Name]\nSeries example\n\n[Data]\nY [ m ],Temperature [ K ]\n0.0,1200\n0.01,1250\n0.02,1220\n"""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "q4.5n4.dat"
            path.write_text(content, encoding="utf-8")
            imported = import_dat_files([path])
        self.assertEqual(len(imported.data), 3)
        self.assertAlmostEqual(float(imported.summary.loc[0, "文件名Q"]), 4.5)
        self.assertAlmostEqual(float(imported.summary.loc[0, "文件名N"]), 4.0)
        self.assertEqual(imported.summary.loc[0, "状态"], "可用")


if __name__ == "__main__":
    unittest.main()
