"""Reproducible analytical and synthetic validation; never reads private research data.

ZDT1 uses two variables, rather than the usual 30-variable benchmark.
The direct-function check isolates NSGA-II; the other checks use run_general.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import scipy
import sklearn
from threadpoolctl import threadpool_limits

from mtpv_optimizer import __version__
from mtpv_optimizer.general import run_general, design_template
from mtpv_optimizer.nsga2 import optimize_nsga2
from mtpv_optimizer.presets import example
from mtpv_optimizer.problem import Constraint, Objective, Response, Problem, Variable, save_project, load_project
from mtpv_optimizer.workflow import export_workbook


SEEDS = list(range(10))
POPULATION, GENERATIONS = 80, 100
LIMITS = {"zdt1_igd_mean": .02, "zdt1_hv_ratio": .95,
          "quadratic_position_error": .02, "mixed_objective_gap": 1e-4,
          "true_constraint_violation": 1e-10, "exact_model_holdout_rmse": 1e-8}


def front2(values):
    """Independent two-objective minimization skyline; exact duplicate vectors collapse."""
    values = np.asarray(values, float)
    order = np.lexsort((values[:, 1], values[:, 0]))
    chosen, best = [], np.inf
    for i in order:
        if values[i, 1] < best:
            chosen.append(i)
            best = values[i, 1]
    return np.asarray(chosen, int)


def hypervolume2(values, reference=(1.1, 1.1)):
    """Union area of dominated rectangles, for two minimized objectives."""
    values = np.asarray(values, float)
    ref = np.asarray(reference, float)
    values = values[np.all(np.isfinite(values), axis=1) & np.all(values < ref, axis=1)]
    if not len(values):
        return 0.0
    points = values[front2(values)]
    total, previous_y = 0., ref[1]
    for x, y in points:
        total += (ref[0] - x) * (previous_y - y)
        previous_y = y
    return float(total)


def igd_mean(reference, candidates):
    """Mean Euclidean reference-to-set distance (IGD with mean aggregation)."""
    return float(cKDTree(np.asarray(candidates, float)).query(np.asarray(reference, float))[0].mean())


def zdt1(x):
    x = np.atleast_2d(x)
    g = 1 + 9 * x[:, 1:].mean(axis=1)
    return np.column_stack([x[:, 0], g * (1 - np.sqrt(x[:, 0] / g))])


class ZDTOracle:
    lower = np.array([0., 0.])
    upper = np.array([1., 1.])
    integer_indices = ()
    target_names = ["f1", "f2"]
    directions = ["min", "min"]
    require_positive = False

    def __init__(self):
        self.evaluations = 0

    def predict(self, x):
        self.evaluations += len(x)
        return zdt1(x)


def quadratic(x):
    x = np.atleast_2d(x)
    return (x[:, 0] - .37)**2 + 2 * (x[:, 1] + .23)**2


def mixed(x):
    x = np.atleast_2d(x)
    loss = (x[:, 0] - .45)**2 + .08 * (x[:, 1] - 2)**2 + .1 * (x[:, 2] - .5)**2
    temperature = 300 + x[:, 0] + .2 * x[:, 1] + x[:, 2]
    return np.column_stack([loss, temperature])


def problem_data(kind):
    rng = np.random.default_rng(20261005)
    if kind == "quadratic":
        train = np.array([(x, y) for x in np.linspace(-1, 1, 6) for y in np.linspace(-1, 1, 6)])
        test = rng.uniform(-1, 1, (20, 2))
        variables = [Variable("x", -1, 1), Variable("y", -1, 1)]
        response = np.r_[quadratic(train), quadratic(test)][:, None]
        columns = ["loss"]
        problem = Problem("Known interior optimum", variables, [Objective("loss", "min")])
    else:
        train = np.array([(x, k, d) for x in np.linspace(-1, 1, 9) for k in range(1, 5) for d in [.1, .5, .9]])
        test = np.column_stack([rng.uniform(-1, 1, 30), rng.integers(1, 5, 30), rng.choice([.1, .5, .9], 30)])
        variables = [Variable("x", -1, 1), Variable("k", 1, 4, kind="integer"),
                     Variable("d", .1, .9, kind="discrete", allowed=[.1, .5, .9])]
        response = mixed(np.vstack([train, test]))
        columns = ["loss", "temperature"]
        problem = Problem("Mixed constrained known optimum", variables, [Objective("loss", "min")],
                          constraints=[Constraint("temperature", ">=", 301.5)], responses=[Response("temperature")])
    frame = pd.DataFrame(np.column_stack([np.vstack([train, test]), response]), columns=[v.column for v in variables] + columns)
    frame["role"] = ["train"] * len(train) + ["test"] * len(test)
    frame.insert(0, "case", [f"ANALYTICAL_{i:03d}" for i in range(len(frame))])
    problem = replace(problem, model="RSM-Quadratic", holdout_column="role", holdout_value="test",
                      population=POPULATION, generations=GENERATIONS, runs=1,
                      description="Analytical software verification; not physical measurements.")
    return problem, frame


def mixed_reference():
    rows = []
    for k in range(1, 5):
        for d in [.1, .5, .9]:
            lower = max(-1., 1.5 - .2 * k - d)
            if lower <= 1:
                x = np.clip(.45, lower, 1.)
                loss, temp = mixed([[x, k, d]])[0]
                rows.append(dict(x=x, k=k, d=d, loss=loss, temperature=temp))
    table = pd.DataFrame(rows).sort_values("loss", kind="stable").reset_index(drop=True)
    return table


def domain_truth(key, frame):
    """Closed synthetic functions, independently reevaluated at returned candidates."""
    if key == "battery":
        f, c, n = frame[["flow_L_min", "channel_mm", "channels"]].to_numpy(float).T
        return np.column_stack([320 - 5*f - .45*c - .5*n + .2*f*f,
                                8 - .7*f - .3*c - .25*n + .04*c*c,
                                .5 + 1.2*f*f + .15*n + .05*c])
    if key == "exchanger":
        v, p = frame[["velocity_m_s", "pitch_mm"]].to_numpy(float).T
        return np.column_stack([200 + 80*v - 15*p + 2*v*p, 20 + 15*v*v + 4*p])
    t, w = frame[["thickness_mm", "width_mm"]].to_numpy(float).T
    return np.column_stack([.002*t*w, 30/(t*t*w), (t*w-100)/100, 10+.08*t*w+.1*w*w])


def serial(value):
    if isinstance(value, (np.integer, np.floating)): return value.item()
    if isinstance(value, np.bool_): return bool(value)
    if isinstance(value, np.ndarray): return value.tolist()
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "validation" / "results")
    args = parser.parse_args()
    folder = args.output.resolve(); folder.mkdir(parents=True, exist_ok=True)
    records, points, checks, outputs = [], [], [], {}
    started = time.perf_counter()
    def check(name, condition, detail):
        checks.append(dict(check=name, passed=bool(condition), detail=detail))

    u = np.linspace(0, 1, 2001)
    reference = np.column_stack([u, 1 - np.sqrt(u)])
    pd.DataFrame(reference, columns=["f1", "f2"]).to_csv(folder / "zdt1_reference.csv", index=False)
    ref_hv = hypervolume2(reference)
    for seed in SEEDS:
        oracle = ZDTOracle()
        result = optimize_nsga2(oracle, POPULATION, GENERATIONS, seed, runs=1, decision_method="IDEAL",
                               anchors=(np.array([0., 0.]), np.array([1., 10.])))
        count = oracle.evaluations
        rng = np.random.default_rng(seed)
        baseline_x = rng.uniform(0, 1, (count, 2))
        baseline_y = zdt1(baseline_x)
        ids = front2(baseline_y)
        for method, x, y in [("NSGA-II", result.variables, result.objectives), ("Random", baseline_x[ids], baseline_y[ids])]:
            row = dict(benchmark="ZDT1-direct-2D", method=method, seed=seed, evaluations=count,
                       points=len(y), igd_mean=igd_mean(reference, y), hv=hypervolume2(y),
                       hv_ratio=hypervolume2(y)/ref_hv)
            records.append(row)
            for a, b in zip(x, y): points.append(dict(method=method, seed=seed, x1=a[0], x2=a[1], f1=b[0], f2=b[1]))
        print(f"ZDT1 seed {seed}: IGD={records[-2]['igd_mean']:.5g}, HV ratio={records[-2]['hv_ratio']:.5g}", flush=True)
    pd.DataFrame(points).to_csv(folder / "zdt1_fronts.csv", index=False)
    zdt = pd.DataFrame(records)
    ours = zdt.loc[zdt.method.eq("NSGA-II")]
    check("ZDT1 known front coverage (all 10 seeds)", ours.igd_mean.le(LIMITS["zdt1_igd_mean"]).all(), "Mean reference-to-set distance <= 0.02; objectives unscaled [0,1] front")
    check("ZDT1 dominated area (all 10 seeds)", ours.hv_ratio.ge(LIMITS["zdt1_hv_ratio"]).all(), "HV / sampled analytical-front HV >= 0.95, reference (1.1,1.1)")

    mixed_ref = mixed_reference(); mixed_ref.to_csv(folder / "mixed_reference.csv", index=False)
    for kind in ("quadratic", "mixed"):
        problem, frame = problem_data(kind)
        frame.to_csv(folder / f"{kind}_data.csv", index=False)
        save_project(folder / f"{kind}.optproj", problem, frame)
        for seed in SEEDS:
            output = run_general(frame, replace(problem, seed=seed))
            selected = output.predicted.iloc[0]
            row = dict(benchmark=kind, method="full surrogate pipeline", seed=seed,
                       train_count=output.config["sample_count"], holdout_count=output.config["holdout_count"],
                       holdout_rmse_max=output.holdout_metrics["留出RMSE"].max(),
                       x=float(selected.x), predicted_loss=float(selected.loss))
            if kind == "quadratic":
                row.update(y=float(selected.y), true_loss=float(quadratic([[selected.x, selected.y]])[0]),
                           position_error=float(np.linalg.norm([selected.x-.37, selected.y+.23])))
            else:
                truth = mixed(output.predicted[["x", "k", "d"]].to_numpy(float))
                row.update(k=float(selected.k), d=float(selected.d), true_loss=float(truth[0, 0]),
                           true_objective_gap=float(truth[0, 0]-mixed_ref.iloc[0].loss),
                           max_true_violation=float(np.maximum(301.5-truth[:, 1], 0).max()),
                           integer_valid=bool(np.all(output.predicted.k.eq(output.predicted.k.round()))),
                           finite_set_valid=bool(output.predicted.d.isin([.1,.5,.9]).all()))
            records.append(row)
            if seed == 0:
                outputs[kind] = output
                output.bundle.holdout_frame.to_csv(folder / f"{kind}_holdout_predictions.csv", index=False)
                output.predicted.to_csv(folder / f"{kind}_candidates.csv", index=False)
                export_workbook(output, folder / f"{kind}_results.xlsx")
            print(f"{kind} seed {seed}: true loss={row['true_loss']:.6g}", flush=True)
        subset = pd.DataFrame([r for r in records if r["benchmark"] == kind])
        check(f"{kind} excluded-holdout prediction", subset.holdout_rmse_max.le(LIMITS["exact_model_holdout_rmse"]).all(), "RSM represents these quadratic/linear functions exactly; not evidence of physical accuracy")
        if kind == "quadratic":
            check("Interior analytical optimum (all 10 seeds)", subset.position_error.le(LIMITS["quadratic_position_error"]).all(), "Euclidean coordinate error <= 0.02 against (0.37,-0.23), absent from training grid")
        else:
            check("Mixed-variable analytical optimum (all 10 seeds)", subset.true_objective_gap.abs().le(LIMITS["mixed_objective_gap"]).all(), "Absolute true loss gap <= 1e-4 against independent finite enumeration")
            check("Mixed-variable ground-truth feasibility", subset.max_true_violation.le(LIMITS["true_constraint_violation"]).all() and subset.integer_valid.all() and subset.finite_set_valid.all(), "Reevaluate original functions, integer k, allowed d, temperature >=301.5")

    domain_rows = []
    for key in ("battery", "exchanger", "structure"):
        problem, frame = example(key)
        output = run_general(frame, problem)
        outputs[key] = output
        prediction = output.predicted[problem.target_names].to_numpy(float)
        truth = domain_truth(key, output.predicted)
        scale = np.ptp(frame[problem.target_names].to_numpy(float), axis=0)
        rmse = np.sqrt(np.mean((prediction-truth)**2, axis=0)) / scale
        if key == "battery": feasible = truth[:, 0] <= 315 + 1e-10
        elif key == "exchanger": feasible = truth[:, 1] <= 350 + 1e-10
        else: feasible = truth[:, 2] >= -1e-10
        row = dict(domain=key, source="synthetic_function", samples=len(frame), variables=len(problem.variables),
                   objectives=len(problem.objectives), candidates=len(prediction),
                   true_feasible_fraction=float(feasible.mean()), worst_candidate_nrmse=float(rmse.max()),
                   max_predicted_constraint_violation=float(output.predicted["约束违反量"].max()))
        domain_rows.append(row)
        check(f"{key} synthetic constraints by function reevaluation", feasible.all(), "Checks the illustrative function, not CFD or experiments")
        record = output.predicted.copy()
        for j, target in enumerate(problem.target_names): record[f"true_{target}"] = truth[:, j]
        record.to_csv(folder / f"{key}_candidates.csv", index=False)
        export_workbook(output, folder / f"{key}_results.xlsx")
        print(f"Domain {key}: true feasibility {row['true_feasible_fraction']:.1%}", flush=True)

    # An adversarial holdout check: change only withheld responses, not training or search.
    p, frame = problem_data("quadratic")
    p = replace(p, population=24, generations=12)
    a = run_general(frame, p)
    changed = frame.copy(); changed.loc[changed.role.eq("test"), "loss"] += 10000
    b = run_general(changed, p)
    identical = np.array_equal(a.predicted[p.variable_names+p.target_names], b.predicted[p.variable_names+p.target_names])
    check("Holdout adversarial isolation", identical and a.config["clean_data_sha256"] == b.config["clean_data_sha256"], "Adding 10000 to all test responses leaves predictions and training checksum identical")
    pd.DataFrame([dict(holdout_shift=0, rmse=a.holdout_metrics["留出RMSE"].max()),
                  dict(holdout_shift=10000, rmse=b.holdout_metrics["留出RMSE"].max())]).to_csv(folder / "holdout_isolation.csv", index=False)
    restored, restored_frame = load_project(folder / "mixed.optproj")
    mp, mf = problem_data("mixed")
    check("Project data and configuration round trip", restored.to_dict() == mp.to_dict() and restored_frame.equals(mf), "Restore raw numerical table and complete configuration")
    design = design_template(mp, 30, 3)
    check("DOE retained center replicates", len(design) == 33 and design.design_role.eq("center_replicate").sum() == 3 and design.d.isin([.1,.5,.9]).all() and design.k.eq(design.k.round()).all(), "30 LHS + 3 center rows, numeric-set and integer validity")
    design.to_csv(folder / "mixed_DOE.csv", index=False)

    pd.DataFrame(records).to_csv(folder / "benchmark_runs.csv", index=False)
    pd.DataFrame(domain_rows).to_csv(folder / "domain_runs.csv", index=False)
    pd.DataFrame(checks).to_csv(folder / "checks.csv", index=False)
    summary = dict(software_version=__version__, seeds=SEEDS, population=POPULATION, generations=GENERATIONS,
                   zdt_dimension=2, thresholds=LIMITS, checks=checks, status="passed" if all(c["passed"] for c in checks) else "failed",
                   environment=dict(python=platform.python_version(), platform=platform.system(), numpy=np.__version__,
                                    pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__),
                   seconds=time.perf_counter()-started,
                   scope="Analytical and synthetic software verification; no private records or physical validation.",
                   sources=["https://pymoo.org/problems/definition.html", "https://pymoo.org/misc/indicators.html"])
    summary["source_hash_encoding"] = "UTF-8 text with LF newlines (platform-independent)"
    summary["source_sha256"] = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_text(encoding="utf-8").encode("utf-8")).hexdigest() for p in [Path(__file__), ROOT/'mtpv_optimizer/nsga2.py', ROOT/'mtpv_optimizer/general.py', ROOT/'mtpv_optimizer/core.py']}
    (folder / "summary.json").write_text(json.dumps(summary, indent=2, default=serial)+"\n", encoding="utf-8")
    make_figures(folder, outputs)
    print(json.dumps({"status":summary["status"], "checks":len(checks), "failed":[c["check"] for c in checks if not c["passed"]]}, ensure_ascii=False), flush=True)
    return 0 if summary["status"] == "passed" else 1


def make_figures(folder, outputs):
    """Evidence-first quantitative figures; all values read from recorded results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mtpv_optimizer.general_figures import draw_general_pareto, draw_general_validation
    assets = ROOT / "docs" / "assets" / "verification"; assets.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family":"sans-serif", "font.sans-serif":["Arial","DejaVu Sans"],
                         "font.size":8, "axes.spines.top":False, "axes.spines.right":False,
                         "axes.linewidth":.7, "legend.frameon":False, "svg.fonttype":"none", "pdf.fonttype":42})
    blue, teal, orange, gray = "#2864a2", "#258f93", "#c77039", "#64748b"
    def save(fig, name):
        fig.savefig(assets / f"{name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(assets / f"{name}.svg", bbox_inches="tight")
        fig.savefig(assets / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    runs = pd.read_csv(folder / "benchmark_runs.csv")
    fronts = pd.read_csv(folder / "zdt1_fronts.csv")
    ref = pd.read_csv(folder / "zdt1_reference.csv")
    fig = plt.figure(figsize=(7.2,3.2), layout="constrained")
    grid=fig.add_gridspec(1,3,width_ratios=[1.6,1,1])
    ax=fig.add_subplot(grid[0,0]); ax.plot(ref.f1,ref.f2,color=gray,lw=1.4,label="Analytical front")
    for method,color in [("Random",orange),("NSGA-II",blue)]:
        f=fronts.loc[fronts.seed.eq(0)&fronts.method.eq(method)]
        ax.scatter(f.f1,f.f2,s=8,color=color,alpha=.8,label=method)
    ax.set(xlabel="f1",ylabel="f2",title="a  ZDT1, two variables; seed 0",xlim=(-.03,1.03),ylim=(-.03,1.1)); ax.legend(fontsize=6.5)
    for j,metric,title in [(1,"igd_mean","b  Front distance ↓"),(2,"hv_ratio","c  Relative hypervolume ↑")]:
        ax=fig.add_subplot(grid[0,j])
        for i,(method,color) in enumerate([("Random",orange),("NSGA-II",blue)]):
            data=runs.loc[runs.benchmark.eq("ZDT1-direct-2D")&runs.method.eq(method),metric].to_numpy()
            ax.scatter(i+np.linspace(-.12,.12,len(data)),data,s=13,color=color)
            ax.hlines(data.mean(),i-.2,i+.2,color="black",lw=1.2)
        ax.set(xticks=[0,1],xticklabels=["Random","NSGA-II"],title=title)
        ax.tick_params(axis='x',rotation=20)
    fig.supxlabel("10 fixed seeds · matched true-function evaluation budgets · lines indicate means",fontsize=7)
    save(fig,"zdt1_benchmark")

    fig, axes = plt.subplots(1,3,figsize=(7.2,2.8),layout="constrained")
    q=runs.loc[runs.benchmark.eq("quadratic")]
    m=runs.loc[runs.benchmark.eq("mixed")]
    axes[0].scatter(q.x,q.y,color=blue,s=23,label="10 seed results")
    axes[0].scatter([.37],[-.23],marker="+",s=95,color=orange,label="Analytical optimum",zorder=4)
    axes[0].set(xlabel="x",ylabel="y",title="a  Interior optimum recovery")
    axes[0].legend(fontsize=6.5)
    axes[1].plot(q.seed,q.position_error,marker="o",ms=3,color=blue)
    axes[1].axhline(LIMITS["quadratic_position_error"],color=gray,ls="--",lw=.8,label="Acceptance limit")
    axes[1].set(xlabel="Seed",ylabel="Coordinate error",title="b  Error across all seeds")
    axes[1].legend(fontsize=6.5)
    axes[2].plot(m.seed,m.true_objective_gap,marker="o",ms=3,color=teal)
    axes[2].axhline(LIMITS["mixed_objective_gap"],color=gray,ls="--",lw=.8)
    axes[2].set(xlabel="Seed",ylabel="True objective gap",title="c  Mixed constrained optimum")
    fig.supxlabel("Full surrogate → constraint → search pipeline; true functions reevaluated after optimization",fontsize=7)
    save(fig,"known_optima")

    test=pd.read_csv(folder / "mixed_holdout_predictions.csv")
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.8),layout="constrained")
    for ax,target,color in zip(axes,["loss","temperature"],[blue,teal]):
        f=test.loc[test["目标"].eq(target)]; truth=f["实测值"]; pred=f["冻结模型预测值"]
        ax.scatter(truth,pred,s=15,color=color)
        ax.plot([truth.min(),truth.max()],[truth.min(),truth.max()],color=gray,ls="--",lw=.8)
        ax.set(xlabel="Held-out function value",ylabel="Frozen model prediction",title=target+" · 30 excluded test designs")
    fig.supxlabel("Representable quadratic/linear functions; near-zero error is a consistency check, not physical accuracy",fontsize=7)
    save(fig,"independent_holdout")

    domains=pd.read_csv(folder / "domain_runs.csv")
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.8),layout="constrained")
    axes[0].bar(domains.domain,100*domains.true_feasible_fraction,color=[blue,teal,gray],width=.55)
    axes[0].set(ylim=(0,110),ylabel="True-function feasible candidates (%)",title="a  Synthetic constraint check")
    for i,row in domains.iterrows(): axes[0].text(i,101,f"n={row.candidates}",ha="center",fontsize=7)
    axes[1].bar(domains.domain,domains.worst_candidate_nrmse,color=[blue,teal,gray],width=.55)
    axes[1].set(ylabel="Worst response NRMSE at candidates",title="b  Surrogate error after reevaluation")
    fig.supxlabel("Three illustrative domains; no experiments or CFD data. Structural GP error remains visible.",fontsize=7)
    save(fig,"domain_checks")
    for key in ("battery","exchanger","structure"):
        problem,_=example(key)
        fig=plt.figure(); draw_general_pareto(fig,outputs[key],problem); save(fig,key+"_pareto")
    p,_=problem_data("mixed")
    fig=plt.figure(); draw_general_validation(fig,outputs["mixed"],p); save(fig,"mixed_validation")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        raise SystemExit(main())
