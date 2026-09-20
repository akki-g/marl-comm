"""Headless, seed-level reporting; absent results remain absent, never fabricated."""

from __future__ import annotations
import csv
import json
from pathlib import Path
import uuid

from commstudy.experiments.mechanism_suite import (
    CONDITIONS,
    latest_attempt,
    manifest,
    read_json,
    verify_plan,
    verify_receipt,
    write_new,
)


def status(root, *, scheduler=False):
    root = Path(root)
    plan = verify_plan(root, check_source=False)
    cells = []
    for task in CONDITIONS:
        for row in manifest(root, task):
            training = latest_attempt(root, row)
            evaluation = latest_attempt(root, row, "evaluation")
            cells.append(
                {k: row[k] for k in ("task", "condition", "method", "seed", "cohort")}
                | {
                    "training": training["status"] if training else "missing",
                    "evaluation": evaluation["status"] if evaluation else "missing",
                }
            )
    states = {}
    if scheduler:
        from commstudy.experiments.mechanism_slurm import scheduler_status

        ledger = root / "submissions.jsonl"
        for line in ledger.read_text().splitlines() if ledger.exists() else []:
            row = json.loads(line)
            if row.get("job_id"):
                try:
                    states[row["job_id"]] = scheduler_status(row["job_id"])
                except FileNotFoundError:
                    states[row["job_id"]] = {"available": False, "reason": "sacct unavailable"}
    return {
        "planned": len(cells),
        "cells": cells,
        "counts": plan["counts"],
        "scheduler": states,
        "scheduler_completion_is_scientific_completion": False,
        "main_approved": (root / "gates/main_approval.json").exists(),
        "complete": all(c["training"] == c["evaluation"] == "completed" for c in cells),
    }


def _csv(path, rows):
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()}
            for row in rows
        )


def analyze(root):
    from commstudy.analysis.mechanism_suite import contrasts

    root = Path(root)
    verify_plan(root)
    output = root / "analysis" / uuid.uuid4().hex
    output.mkdir(parents=True, exist_ok=False)
    completion = status(root)
    outcomes, parameters, costs, curves, physical, inputs = [], [], [], [], [], {}
    native_validation = []
    for task in CONDITIONS:
        for row in manifest(root, task):
            identity = {k: row[k] for k in ("task", "condition", "method", "seed", "cohort")}
            trained = latest_attempt(root, row)
            if trained and trained["status"] == "completed":
                verify_receipt(root, trained)
                inputs.update(trained["artifacts"])
                parameters.append(identity | trained["parameter_counts"])
                summary = read_json(trained["training_summary_path"])
                metrics_path = Path(summary["run_dir"]) / "metrics.csv"
                if metrics_path.exists():
                    with metrics_path.open(newline="") as stream:
                        native_validation.extend(
                            identity | point
                            for point in csv.DictReader(stream)
                            if point["phase"].startswith("evaluation")
                        )
                for point in trained["validation"]:
                    summaries = point["summaries"]["live"]
                    value = (
                        summaries["contact_by_deadline"]
                        if task == "pcp"
                        else summaries["domain_means"]["voltage_violation_magnitude"]
                    )
                    curves.append(identity | {"frames": point["frames"], "primary": value})
            receipt = latest_attempt(root, row, "evaluation")
            if not receipt or receipt["status"] != "completed":
                continue
            verify_receipt(root, receipt)
            inputs.update(receipt["artifacts"])
            result = read_json(receipt["result_path"])
            if task == "pcp":
                live = result["summaries"]["live"]["contact_by_deadline"]
                severed = result["summaries"]["severed"]["contact_by_deadline"]
                valid = True
                for arm, episodes in result["arms"].items():
                    for episode in episodes:
                        costs.append(
                            identity
                            | {"arm": arm, "episode_id": episode["episode_id"]}
                            | episode["module_stats"]
                        )
            else:
                live = result["summaries"]["live"]["domain_means"]["voltage_violation_magnitude"]
                severed = result["summaries"]["severed"]["domain_means"][
                    "voltage_violation_magnitude"
                ]
                valid = all(
                    s["solver_failure_count"] == 0
                    and s["unattempted_transition_count"] == 0
                    and s["voltage_observed_transitions"] == s["planned_transition_count"]
                    for s in result["summaries"].values()
                )
                for arm, summary in result["summaries"].items():
                    actions = [
                        float(action[0])
                        for episode in result["arms"][arm]
                        for step in episode["transitions"]
                        for action in step["actions"]
                    ]
                    saturated = sum(abs(value) > 0.999 for value in actions)
                    physical.append(
                        identity
                        | {"arm": arm}
                        | summary
                        | {
                            "action_scalar_count": len(actions),
                            "saturated_action_scalars": saturated,
                            "action_saturation_fraction": saturated / len(actions)
                            if actions
                            else None,
                        }
                    )
                    for episode in result["arms"][arm]:
                        for module in episode["communication_stats_by_module"]:
                            costs.append(
                                identity
                                | {"arm": arm, "episode_id": episode["episode_id"]}
                                | module
                            )
            outcomes.append(identity | {"live": live, "severed": severed, "valid": valid})
    doc = read_json(root / "preparation/suite.json")
    stats = contrasts(outcomes, doc["seeds"], doc["bootstrap_seed"])
    complete = completion["complete"] and all(r["valid"] for r in outcomes)
    report = {
        "scientific_complete": complete,
        "completion": completion,
        "statistics": stats,
        "outcomes": outcomes,
        "inputs": inputs,
        "planned_seed_denominator": len(doc["seeds"]),
        "interpretation": (
            "Fixed-budget adaptations. Effects may be negative/inconclusive. "
            "Reliance is an input-shift intervention."
        ),
        "physical_scope": (
            "Voltage statistics conditional on valid solves; failed banks prevent clean rankings."
        ),
        "figures": "No outcome plots without actual retained outcomes",
        "output": str(output),
    }
    for name, rows in [
        ("completion", completion["cells"]),
        ("policy_outcomes", outcomes),
        ("parameters", parameters),
        ("payload", costs),
        ("validation_curves", curves),
        ("native_validation_complete", native_validation),
        ("mapdn_physical", physical),
        ("effects", stats["effects"]),
        ("reliance", stats["reliance"]),
        ("visibility_interactions", stats["visibility_interactions"]),
        ("supplement_effects", stats["supplement_effects"]),
    ]:
        _csv(output / f"{name}.csv", rows)
    if outcomes:
        _plots(output, outcomes, stats, curves, physical)
        report["figures"] = sorted(p.name for p in output.glob("*.png"))
    if native_validation:
        _native_curve_plot(output, native_validation)
    write_new(output / "report.json", report)
    (output / "README.md").write_text(
        f"# Communication mechanisms v1\n\nScientific completion: **{complete}**.\n\n"
        f"Planned policies: {completion['planned']}; evaluated policy records: {len(outcomes)}. "
        "Missing/failed rows remain in completion.csv. Episodes are averaged within each policy; "
        "intervals resample training seed blocks (10,000 PCG64 draws), are descriptive and "
        "pointwise, "
        "and do not adjust for 12 primary comparisons. Partial cells have complete=false.\n\n"
        "PCP effects are absolute probabilities; MAPDN voltage effects are p.u. No "
        "cross-task raw ranking. "
        "Soft gates do not save packets; payload excludes headers. LocalCapacity32 matches "
        "Broadcast "
        "projection counts, not Attention/Graph. Frozen-policy reliance includes "
        "distribution shift.\n\n"
        "See source documentation docs/experiments/pcp_mapdn_mechanisms_v1/README.md and "
        "docs/RESEARCH_REFERENCES.md for adaptation and interpretation references.\n"
    )
    return report


def _plots(output, outcomes, stats, curves, physical):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for task, conditions in CONDITIONS.items():
        for condition in conditions:
            selected = [
                r
                for r in outcomes
                if r["task"] == task and r["condition"] == condition and r["valid"]
            ]
            if not selected:
                continue
            methods = list(dict.fromkeys(r["method"] for r in selected))
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            for i, method in enumerate(methods):
                values = [r for r in selected if r["method"] == method]
                axes[0].scatter([i] * len(values), [r["live"] for r in values], alpha=0.7)
                effects = [
                    e
                    for e in stats["effects"]
                    if e["task"] == task
                    and e["condition"] == condition
                    and e["method"] == method
                    and e["primary"]
                    and e["estimate"]
                ]
                for effect in effects:
                    estimate = effect["estimate"]
                    axes[1].scatter([i] * estimate["n"], estimate["seed_differences"], alpha=0.7)
                    if estimate["ci95"]:
                        axes[1].plot([i, i], estimate["ci95"], color="black")
                    paired = []
                    for row in values:
                        base = next(
                            (
                                r
                                for r in selected
                                if r["method"] == "identity" and r["seed"] == row["seed"]
                            ),
                            None,
                        )
                        if base:
                            sign = 1 if task == "pcp" else -1
                            paired.append(
                                (
                                    sign * (row["live"] - base["live"]),
                                    sign * (row["live"] - row["severed"]),
                                )
                            )
                    if paired:
                        axes[2].scatter(*zip(*paired, strict=True), label=method)
            for ax in axes[:2]:
                ax.set_xticks(range(len(methods)), methods, rotation=45, ha="right")
            axes[0].set_ylabel(
                "Contact by transition 50" if task == "pcp" else "Voltage violation (p.u.)"
            )
            axes[1].set_ylabel("Benefit versus Identity (native units)")
            axes[1].axhline(0, color="gray")
            axes[2].set(xlabel="Benefit versus Identity", ylabel="Live/severed reliance")
            axes[2].axhline(0, color="gray")
            if axes[2].collections:
                axes[2].legend()
            fig.suptitle(f"{task} / {condition}: individual seeds; incomplete cells labeled in CSV")
            fig.tight_layout()
            fig.savefig(output / f"{task}_{condition}_outcomes.png", dpi=160)
            plt.close(fig)
    if curves:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, (task, condition) in zip(
            axes, [("pcp", "radius1"), ("pcp", "global"), ("mapdn", "case33")], strict=True
        ):
            for method in dict.fromkeys(r["method"] for r in curves):
                for seed in dict.fromkeys(r["seed"] for r in curves):
                    points = [
                        r
                        for r in curves
                        if r["task"] == task
                        and r["condition"] == condition
                        and r["method"] == method
                        and r["seed"] == seed
                    ]
                    if points:
                        ax.plot(
                            [r["frames"] for r in points],
                            [r["primary"] for r in points],
                            alpha=0.55,
                            label=f"{method}/{seed}",
                        )
            if ax.lines:
                ax.legend(fontsize=5, ncol=2)
            ax.set(
                title=f"{task}/{condition}",
                xlabel="Collected training frames",
                ylabel="Validation primary (native units)",
            )
        fig.tight_layout()
        fig.savefig(output / "validation_curves.png", dpi=160)
        plt.close(fig)
    live = [r for r in physical if r["arm"] == "live"]
    if live:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for r in live:
            means = r["domain_means"]
            for ax, metric, label in zip(
                axes,
                ["q_loss", "total_line_loss"],
                ["Reactive effort (MVAr)", "Line loss (MW)"],
                strict=True,
            ):
                if means.get(metric) is not None:
                    ax.scatter(
                        means[metric],
                        means["voltage_violation_magnitude"],
                        label=f"{r['method']}/{r['seed']}",
                    )
                ax.set(xlabel=label, ylabel="Observed voltage violation (p.u.)")
        fig.tight_layout()
        fig.savefig(output / "mapdn_tradeoffs.png", dpi=160)
        plt.close(fig)


def _native_curve_plot(output, records):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, condition in zip(axes, ("radius1", "global"), strict=True):
        selected = [
            r
            for r in records
            if r["task"] == "pcp"
            and r["condition"] == condition
            and r["phase"] == "evaluation"
            and r["group"] == "adversary"
            and r["metric"] == "return_mean"
        ]
        for method, seed in sorted({(r["method"], r["seed"]) for r in selected}):
            points = sorted(
                (int(r["frames"]), float(r["value"]))
                for r in selected
                if r["method"] == method and r["seed"] == seed
            )
            ax.plot(
                [p[0] for p in points], [p[1] for p in points], alpha=0.6, label=f"{method}/{seed}"
            )
        if ax.lines:
            ax.legend(fontsize=5, ncol=2)
        ax.axvline(600000, color="gray", linestyle="--")
        ax.set(
            title=f"PCP {condition}: complete native validation cadence",
            xlabel="Training frames",
            ylabel="Predator-group return",
        )
    fig.tight_layout()
    fig.savefig(output / "complete_native_validation.png", dpi=160)
    plt.close(fig)
