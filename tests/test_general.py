from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from mtpv_optimizer.core import decision_scores, ideal_distances
from mtpv_optimizer.general import run_general, validation_splits, _clean, design_template
from mtpv_optimizer.nsga2 import _non_dominated_sort
from mtpv_optimizer.presets import example
from mtpv_optimizer.problem import Arithmetic, Constraint, ConstraintEvaluator, Objective, Response, Problem, Variable, load_project, save_project
from mtpv_optimizer.workflow import export_workbook


def small(problem):
    return replace(problem, population=24, generations=12, runs=1)


class GeneralTests(unittest.TestCase):
    def setUp(self):
        x = np.linspace(-1, 1, 25)
        self.frame = pd.DataFrame({"x": x, "loss": x, "benefit": -x, "batch": np.repeat(np.arange(5), 5), "time": np.arange(25)})
        self.problem = Problem("signed responses", [Variable("x", -1, 1)],
            [Objective("loss", "min"), Objective("benefit", "max")], model="Linear", population=24, generations=12, runs=1)

    def test_known_optimum_with_signed_and_zero_responses(self):
        output = run_general(self.frame, self.problem)
        self.assertAlmostEqual(output.observed.iloc[0]["x"], -1)
        self.assertAlmostEqual(output.predicted.iloc[0]["x"], -1)
        self.assertTrue(all(m.r2_cv > .999999 for m in output.bundle.metrics))
        self.assertTrue(all(np.isnan(m.mape_cv_pct) for m in output.bundle.metrics))

    def test_single_objective_and_one_dimensional_hull(self):
        p = replace(self.problem, objectives=[Objective("benefit", "min")], domain="hull")
        output = run_general(self.frame, p)
        self.assertAlmostEqual(output.predicted.iloc[0]["x"], 1)
        self.assertTrue(output.bundle.domain.contains(output.predicted[["x"]].to_numpy()).all())

    def test_arbitrary_directions_and_decision_methods(self):
        values = np.array([[0, 4, -3, 8], [1, 3, -2, 7]])
        # Row zero dominates across max/min/max/min directions after reversing rows.
        directions = ["min", "max", "min", "max"]
        for method in ("IDEAL", "TOPSIS", "ARAS"):
            scores = decision_scores(values, [1] * 4, method, (values.min(axis=0), values.max(axis=0)), directions)
            self.assertGreater(scores[0], scores[1])
        distances = ideal_distances([[3, -2]], [1, 1], ([0, 0], [1, 1]), ["max", "min"])
        self.assertEqual(distances[0], 0, "Beyond-ideal improvement must not become a loss")

    def test_feasible_first_dominance(self):
        fitness = np.array([[0, 0], [10, 10], [-1, -1], [99, 99]])
        fronts, _ = _non_dominated_sort(fitness, [2, 0, 3, 1])
        self.assertEqual([front.tolist() for front in fronts], [[1], [3], [0], [2]])

    def test_response_and_relationship_constraints_drive_search(self):
        p = replace(self.problem, constraints=[Constraint("x", ">=", .25), Constraint("loss + benefit", "==", 0, tolerance=1e-8)])
        output = run_general(self.frame, p)
        self.assertTrue((output.predicted["x"] >= .25 - 1e-10).all())
        self.assertLess(float(output.predicted.iloc[0]["x"]), .3)
        self.assertTrue((output.predicted["约束违反量"] <= 1e-10).all())
        self.assertTrue((output.constraint_audit["可行"] == (output.constraint_audit["x"] >= .25 - 1e-10)).all())

    def test_no_feasible_design_reports_failure(self):
        p = replace(self.problem, constraints=[Constraint("loss", "<=", -10)])
        with self.assertRaisesRegex(ValueError, "没有找到满足"):
            run_general(self.frame, p)

    def test_constraint_penalties_can_be_augmented_with_copy_on_write(self):
        p = replace(self.problem, constraints=[Constraint("x", ">=", 0)])
        evaluator = ConstraintEvaluator(p)
        x = self.frame[["x"]].to_numpy()
        y = self.frame[["loss", "benefit"]].to_numpy()
        with pd.option_context("mode.copy_on_write", True):
            penalties = evaluator(x, y)
            penalties[0] = np.inf
            penalties += .25  # Search adds nonfinite and coverage penalties in place.
            fresh = evaluator(x, y)
        self.assertEqual(fresh[0], 1.0)
        self.assertEqual(fresh[-1], 0.0)

    def test_nearest_sample_limit(self):
        p = replace(self.problem, max_distance=.0005, constraints=[Constraint("x", ">=", 0)])
        output = run_general(self.frame, p)
        self.assertTrue((output.predicted["最近样本距离"] <= .0005 + 1e-10).all())

    def test_two_other_domains_without_source_changes(self):
        for key in ("battery", "exchanger"):
            problem, frame = example(key)
            output = run_general(frame, small(problem))
            self.assertFalse(output.predicted.empty)
            self.assertTrue((output.predicted["约束违反量"] <= 1e-10).all())
            self.assertEqual(output.predicted["结果类型"].unique().tolist(), ["代理模型预测"])
            if key == "battery":
                np.testing.assert_allclose(output.predicted["channels"], np.rint(output.predicted["channels"]))
            else:
                self.assertTrue(output.predicted["pitch_mm"].isin([2, 4, 6]).all())
            self.assertIn("合成", output.config["description"])

    def test_four_objectives_and_negative_safety_margin(self):
        problem, frame = example("structure")
        output = run_general(frame, small(problem))
        self.assertEqual(len(output.bundle.metrics), 4)
        self.assertTrue((output.predicted["safety_margin"] >= -1e-10).all())
        self.assertEqual(output.config["directions"], ["min", "min", "max", "min"])

    def test_constraint_only_response_is_not_an_extra_objective(self):
        frame = self.frame.assign(temperature=self.frame.x + 300)
        p = replace(self.problem, responses=[Response("temperature", "K")], constraints=[Constraint("temperature", ">=", 300.25)],
                    holdout_column="batch", holdout_value="2")
        output = run_general(frame, p)
        self.assertEqual(output.bundle.target_names, ["loss", "benefit"])
        self.assertEqual(len(output.bundle.metrics), 3)
        self.assertTrue((output.predicted.temperature >= 300.25 - 1e-9).all())
        self.assertLess(output.predicted.iloc[0].x, .3)
        observed = output.constraint_audit
        self.assertTrue((observed["可行"] == observed.temperature.ge(300.25)).all())
        self.assertIn("temperature", output.bundle.holdout_frame["目标"].unique())

    def test_fractional_discrete_set_with_hull(self):
        frame = pd.DataFrame([(a, b, a + b) for a in [.1, .5, .9] for b in np.linspace(0, 1, 6)], columns=["a", "b", "response"])
        p = Problem("fractional set", [Variable("a", .1, .9, kind="discrete", allowed=[.1, .5, .9]), Variable("b", 0, 1)],
                    [Objective("response", "max")], domain="hull", model="Linear", population=24, generations=12, runs=1)
        output = run_general(frame, p)
        self.assertTrue(output.predicted.a.isin([.1, .5, .9]).all())
        self.assertTrue(output.bundle.domain.contains(output.predicted[["a", "b"]].to_numpy()).all())

    def test_cleaning_and_training_bounds_are_auditable(self):
        bad = self.frame.iloc[[0]].copy(); bad["loss"] = np.inf
        frame = pd.concat([self.frame, bad], ignore_index=True)
        output = run_general(frame, self.problem)
        self.assertEqual(output.cleaning_audit["_source_row"].tolist(), [26])
        p = replace(self.problem, variables=[Variable("x", -2, 1)])
        with self.assertRaisesRegex(ValueError, "外推"):
            run_general(frame, p)
        numerical = replace(self.problem, variables=[Variable("x", -1 - 1e-12, 1 + 1e-12)])
        snapped = run_general(self.frame, numerical)
        self.assertEqual(snapped.config["search_lower"], [-1])
        self.assertEqual(snapped.config["search_upper"], [1])

    def test_holdout_never_influences_selection_or_fit(self):
        holdout = pd.DataFrame({"x": [-.71, .13, .71], "loss": [-.71, .13, .71], "benefit": [.71, -.13, -.71], "batch": "test", "time": [26, 27, 28]})
        frame = pd.concat([self.frame, holdout], ignore_index=True)
        p = replace(self.problem, holdout_column="batch", holdout_value="test")
        a = run_general(frame, p)
        frame.loc[frame.batch.eq("test"), ["loss", "benefit"]] = 10000
        b = run_general(frame, p)
        np.testing.assert_array_equal(a.predicted[self.problem.variable_names + self.problem.target_names], b.predicted[self.problem.variable_names + self.problem.target_names])
        pd.testing.assert_frame_equal(a.bundle.candidate_metrics, b.bundle.candidate_metrics)
        self.assertEqual(a.config["clean_data_sha256"], b.config["clean_data_sha256"])
        self.assertEqual(a.config["holdout_count"], 3)
        self.assertGreater(b.holdout_metrics["留出RMSE"].min(), 9000)

    def test_group_folds_separate_groups(self):
        p = replace(self.problem, validation="group", group_column="batch")
        train, folds, _ = validation_splits(self.frame, p)
        for a, b in folds:
            self.assertFalse(set(train.iloc[a].batch) & set(train.iloc[b].batch))
        output = run_general(self.frame, p)
        self.assertEqual(output.config["cv_folds"], 5)

    def test_time_folds_are_forward_and_initial_block_unscored(self):
        frame = pd.concat([self.frame, self.frame.iloc[[5]]], ignore_index=True).sample(frac=1, random_state=5)
        p = replace(self.problem, validation="time", time_column="time", duplicates="keep")
        train, folds, _ = validation_splits(frame, p)
        for a, b in folds:
            self.assertLess(train.iloc[a].time.max(), train.iloc[b].time.min())
        output = run_general(frame, p)
        validation = output.bundle.validation_frame
        self.assertTrue(validation.loc[validation.fold.eq(0), "交叉验证预测值"].isna().all())
        self.assertTrue(validation.loc[validation.fold.gt(0), "交叉验证预测值"].notna().all())
        self.assertTrue(all(m.r2_cv > .999999 for m in output.bundle.metrics))

    def test_replicates_are_grouped_and_mean_is_explicit(self):
        duplicate = self.frame.iloc[[0]].copy(); duplicate["loss"] += .1
        frame = pd.concat([self.frame, duplicate], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "不同响应"):
            run_general(frame, self.problem)
        p = replace(self.problem, duplicates="keep")
        clean, _, _ = _clean(frame, p, [])
        train, folds, _ = validation_splits(clean, p)
        for a, b in folds:
            self.assertFalse(set(train.iloc[a].x) & set(train.iloc[b].x))
        mean, _, _ = _clean(frame, replace(p, duplicates="mean"), [])
        self.assertEqual(len(mean), 25)
        self.assertAlmostEqual(mean.loc[mean.x.eq(-1), "loss"].iloc[0], -.95)
        self.assertEqual(mean.loc[mean.x.eq(-1), "_replicate_count"].iloc[0], 2)

    def test_project_round_trip_restores_data_and_all_settings(self):
        p, frame = example("exchanger")
        p = replace(p, constraints=p.constraints + [Constraint('abs(col("heat_W") - 400)', "<=", 500)], validation="group", group_column="batch", max_distance=.4)
        frame["tiny"] = 1.2345678901234567e-20
        with TemporaryDirectory() as folder:
            path = Path(folder) / "demo.optproj"
            save_project(path, p, frame)
            restored, table = load_project(path)
            self.assertEqual(restored.to_dict(), p.to_dict())
            pd.testing.assert_frame_equal(frame, table)

    def test_safe_constraint_language_rejects_code(self):
        for bad in ('__import__("os").system("echo injected")', 'x.__class__', '[x for x in [1]]', 'open("test")', '(lambda: 1)()'):
            with self.assertRaises(ValueError): Arithmetic(bad, ["x"])
        p = replace(self.problem, constraints=[Constraint('abs(col("loss") + benefit)', "==", 0, tolerance=1e-8)])
        self.assertTrue((ConstraintEvaluator(p)(self.frame[["x"]].to_numpy(), self.frame[["loss", "benefit"]].to_numpy()) == 0).all())

    def test_doe_preserves_centres_and_discrete_values(self):
        problem, _ = example("exchanger")
        design = design_template(problem, 20, 4)
        self.assertEqual(len(design), 24)
        self.assertEqual(design.design_role.eq("center_replicate").sum(), 4)
        self.assertEqual(len(design.loc[design.design_role.eq("center_replicate")].drop_duplicates(problem.variable_names)), 1)
        self.assertTrue(design.pitch_mm.isin([2, 4, 6]).all())

    def test_export_contains_constraints_cleaning_holdout_and_snapshot(self):
        output = run_general(self.frame, self.problem)
        with TemporaryDirectory() as folder:
            path = Path(folder) / "result.xlsx"
            export_workbook(output, path)
            with pd.ExcelFile(path) as excel:
                self.assertTrue({"可行性与覆盖诊断", "数据清理记录", "独立留出指标", "运行设置"}.issubset(excel.sheet_names))
                predicted = pd.read_excel(excel, sheet_name="模型预测Pareto")
                self.assertEqual(predicted["结果类型"].unique().tolist(), ["代理模型预测"])


if __name__ == "__main__":
    unittest.main()
