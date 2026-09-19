"""Plot descriptive PCP confirmation evidence, retaining every training seed.

Example:
    python scripts/plot_pcp_confirmation.py --report results/confirmation.json \
        --out-dir results/confirmation_figures

The report is authoritative. Raw curves are read only when their saved hashes
match the report's artifact bindings. No training, approval, episode pooling,
smoothing, confidence interval, or convergence inference is performed here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, PercentFormatter


FIRST_FRAME, HORIZON = 6000, 600000
OUTCOMES = (
    ("evaluation", "return_mean", "Predator return", "Mean episode return"),
    ("evaluation_domain", "any_contact", "Contact probability", "Fraction of episodes"),
    ("evaluation_domain", "contact_duration_steps", "Contact duration", "Mean steps per episode"),
    ("evaluation_domain", "any_simultaneous_contact", "Simultaneous contact probability",
     "Fraction of episodes"),
    ("evaluation_domain", "predator_boundary_fraction", "Predator boundary occupancy",
     "Mean fraction of predators"),
    ("evaluation_domain", "prey_boundary_fraction", "Prey boundary occupancy",
     "Mean fraction of prey"),
)
HEALTH = (
    ("training", "entropy", "Predator policy entropy", "Transformed-policy entropy"),
    ("training", "loss_critic", "Critic loss", "Loss"),
    ("training", "explained_variance", "Critic explained variance", "Explained variance"),
    ("training", "grad_norm_loss_objective", "Actor gradient norm", "Pre-clipping norm"),
    ("training", "grad_norm_loss_critic", "Critic gradient norm", "Pre-clipping norm"),
    ("training", "kl_approx", "Approximate policy KL", "Upstream minibatch mean"),
    ("collection", "scale_q99", "Raw action scale", "99th percentile"),
    ("collection", "loc_q99", "Raw action location", "Signed 99th percentile"),
    ("collection", "action_saturated_fraction", "Behaviour action saturation",
     "Action scalars with |action| > 0.999 (%)"),
)
EXTRA_CURVES = (
    ("evaluation_domain", "simultaneous_contact_steps"),
    ("evaluation_domain", "episode_count"),
    ("evaluation_domain", "complete_episodes"),
    ("evaluation_domain", "transition_count"),
    ("evaluation_domain", "reward_accounting_max_abs_error"),
    ("collection", "action_finite_fraction"),
    ("collection", "loc_finite_fraction"),
    ("collection", "scale_finite_fraction"),
    ("collection", "replay_log_prob_max_abs_error"),
)
SELECTED = {(phase, metric) for phase, metric, *_ in (*OUTCOMES, *HEALTH)} | set(EXTRA_CURVES)
OUTPUT_NAMES = (
    "per_seed.csv", "curves.csv", "provenance.json",
    "pcp_outcomes.pdf", "pcp_outcomes.png", "pcp_health.pdf", "pcp_health.png",
)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path):
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{path}: expected a JSON object.")
    return document


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _report_curve(run):
    performance = run.get("performance") or {}
    frames = performance.get("evaluation_frames", [])
    values = performance.get("evaluation_returns", [])
    if len(frames) != len(values) or len(set(frames)) != len(frames):
        raise ValueError(f"{run.get('run_id')}: report return curve has inconsistent frames.")
    if any(type(frame) is not int or not FIRST_FRAME <= frame <= HORIZON for frame in frames):
        raise ValueError(f"{run.get('run_id')}: report frame is outside the fixed 6k–600k budget.")
    return {frame: float(value) if _number(value) else math.nan
            for frame, value in zip(frames, values, strict=True)}


def _record(run, report, suite):
    record = {
        "run": run, "metadata": {}, "curves": defaultdict(dict), "issues": [],
        "artifacts": {}, "raw_curves_accepted": False,
    }
    report_returns = _report_curve(run)
    record["curves"]["evaluation", "return_mean"] = report_returns
    run_id = run.get("run_id")
    if not run_id:
        record["issues"].append("No managed run was reported for this expected training seed.")
        return record
    if not isinstance(run_id, str) or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError(f"Unsafe run_id in report: {run_id!r}.")
    if suite is None:
        record["issues"].append(
            "Suite path unavailable; only the report return curve is available."
        )
        return record
    run_dir = suite / run_id
    # Resolve symlinks as well as textual paths before admitting raw artifacts.
    if not run_dir.resolve().is_relative_to(suite.resolve()):
        raise ValueError(f"Run path escapes the suite: {run_dir}.")
    bindings = run.get("artifacts") or {}
    accepted = True
    for name in ("metadata.json", "status.json", "metrics.csv"):
        path = run_dir / name
        actual = _sha256(path) if path.is_file() else None
        expected = bindings.get(name)
        matched = actual is not None and actual == expected
        record["artifacts"][name] = {
            "path": str(path), "sha256": actual, "report_sha256": expected, "matches": matched,
        }
        if not matched:
            accepted = False
            record["issues"].append(
                f"{name}: missing or not bound to this report; raw curves omitted."
            )
    if not accepted:
        return record
    try:
        metadata = _read_json(run_dir / "metadata.json")
        status = _read_json(run_dir / "status.json")
    except (OSError, ValueError) as error:
        record["issues"].append(f"Managed JSON unreadable; raw curves omitted: {error}")
        return record
    record["metadata"] = metadata
    record["managed_status"] = status.get("status", "unknown")
    if metadata.get("seed") != run.get("seed") or metadata.get("run_id") != run_id:
        record["issues"].append("Managed run identity differs from report; raw curves omitted.")
        return record
    if metadata.get("source_sha256") != report.get("source_sha256"):
        record["issues"].append("Training source differs from report; raw curves omitted.")
        return record
    curves = defaultdict(dict)
    duplicate = set()
    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"phase", "group", "metric", "frames", "sample", "value"}
        if not required.issubset(reader.fieldnames or []):
            record["issues"].append("Metrics CSV schema is incomplete; raw curves omitted.")
            return record
        for line, row in enumerate(reader, start=2):
            key = row["phase"], row["metric"]
            if row["group"] != "adversary" or row["sample"] or key not in SELECTED:
                continue
            try:
                frame = int(row["frames"])
            except (ValueError, TypeError):
                record["issues"].append(f"CSV line {line}: invalid frame omitted.")
                continue
            try:
                value = float(row["value"])
            except (ValueError, TypeError):
                value = math.nan
            if not FIRST_FRAME <= frame <= HORIZON:
                record["issues"].append(f"{key}: out-of-budget frame {frame} omitted.")
                continue
            if frame in curves[key]:
                duplicate.add((*key, frame))
                value = math.nan
            curves[key][frame] = value
    # Duplicate observations never get an implicit last-row-wins interpretation.
    for phase, metric, frame in duplicate:
        curves[phase, metric][frame] = math.nan
    if duplicate:
        record["issues"].append(f"{len(duplicate)} duplicate metric/frame keys plotted as gaps.")
    nonfinite = sum(not math.isfinite(v) for curve in curves.values() for v in curve.values())
    if nonfinite:
        record["issues"].append(f"{nonfinite} non-finite selected observations plotted as gaps.")
    raw_returns = curves.get(("evaluation", "return_mean"), {})
    if report_returns and (set(raw_returns) != set(report_returns) or any(
        not math.isclose(raw_returns[frame], value, rel_tol=1e-6, abs_tol=1e-5)
        for frame, value in report_returns.items()
    )):
        record["issues"].append("Raw/report return curves disagree; raw curves omitted.")
        return record
    record["curves"].update(curves)
    record["raw_curves_accepted"] = True
    missing = [
        f"{phase}/{metric}" for phase, metric, *_ in (*OUTCOMES, *HEALTH)
        if not any(math.isfinite(value) for value in curves.get((phase, metric), {}).values())
    ]
    if missing:
        record["issues"].append(
            f"Requested panels without finite observations: {', '.join(missing)}."
        )
    return record


def _seed_row(record, report):
    run, metadata = record["run"], record["metadata"]
    curve = record["curves"].get(("evaluation", "return_mean"), {})
    expected = set(report.get("expected_evaluation_frames", []))
    row = {
        "seed": run.get("seed"), "run_id": run.get("run_id", ""),
        "managed_status": record.get("managed_status", "missing_or_unverified"),
        "validation_passed": run.get("validation_passed", False),
        "raw_curves_accepted": record["raw_curves_accepted"],
        "metadata_frames": metadata.get("frames"), "evaluation_points": len(curve),
        "first_evaluation_frames": min(curve) if curve else None,
        "last_evaluation_frames": max(curve) if curve else None,
        "full_600k_return_coverage": bool(expected) and set(curve) == expected
        and max(curve) == HORIZON and all(math.isfinite(v) for v in curve.values()),
        "report_checks_passed": report.get("checks_passed", False),
        "review_status": report.get("review_status", "unknown"),
        "protocol_id": report.get("protocol_id"),
        "protocol_sha256": report.get("protocol_sha256"),
        "source_sha256": report.get("source_sha256"),
        "runtime_device": metadata.get("runtime", {}).get("train_device"),
        "issues": json.dumps([*run.get("issues", []), *record["issues"]], sort_keys=True),
    }
    row["performance_scope"] = (
        "complete_600k" if row["full_600k_return_coverage"]
        else "observed_prefix_only; no final 600k measurement"
    )
    for phase in ("training", "collection"):
        frames = [frame for (p, _), values in record["curves"].items()
                  for frame in values if p == phase]
        row[f"last_logged_{phase}_frames"] = max(frames) if frames else None
    window = (run.get("performance") or {}).get("final_window_frames", [])
    row["reported_final_window_start_frames"] = min(window) if window else None
    row["reported_final_window_end_frames"] = max(window) if window else None
    for key, value in (run.get("performance") or {}).items():
        if not isinstance(value, (dict, list)):
            row[key] = value if not isinstance(value, float) or math.isfinite(value) else None
    for metric, summary in (run.get("domain_learning_summary") or {}).items():
        for key, value in summary.items():
            row[f"domain_{metric}_{key}"] = value
    for (phase, metric), values in record["curves"].items():
        finite = [value for _, value in sorted(values.items()) if math.isfinite(value)]
        prefix = f"observed_{phase}_{metric}"
        row[f"{prefix}_finite_points"] = len(finite)
        if finite:
            row[f"{prefix}_last"] = finite[-1]
    return row


def _write_csv(path, rows, *, columns=None):
    fields = columns or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _figures(records, report, out_dir, report_hash, review=None):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    })
    devices = {r["metadata"].get("runtime", {}).get("train_device") for r in records}
    devices.discard(None)
    runtime = "/".join(sorted(str(d).upper() for d in devices)) or "Runtime unavailable"
    colors = plt.get_cmap("tab10")
    expected_eval = report.get("expected_evaluation_frames") or [
        FIRST_FRAME, *range(12000, HORIZON + 1, 12000),
    ]
    for name, panels, shape in (("pcp_outcomes", OUTCOMES, (2, 3)),
                                ("pcp_health", HEALTH, (3, 3))):
        fig, axes = plt.subplots(*shape, figsize=(11.7, 7.5 if shape[0] == 2 else 10))
        for ax, (phase, metric, title, ylabel) in zip(axes.flat, panels, strict=True):
            shown = False
            for index, record in enumerate(records):
                values = record["curves"].get((phase, metric), {})
                frames = sorted(set(values) | set(expected_eval if phase.startswith("evaluation")
                                                 else range(FIRST_FRAME, HORIZON + 1, FIRST_FRAME)))
                finite = any(math.isfinite(value) for value in values.values())
                passed = record["run"].get("validation_passed", False)
                ax.plot(frames, [values.get(frame, math.nan) for frame in frames],
                        color=colors(index % 10), linestyle="-" if passed else "--",
                        linewidth=1.15, marker="o", markersize=2.1, markevery=1)
                shown |= finite
            if not shown:
                ax.text(0.5, 0.5, "No verified observations", transform=ax.transAxes,
                        ha="center", va="center", color="0.45")
            ax.set(title=title, ylabel=ylabel, xlabel="Training frames")
            ax.set_xlim(FIRST_FRAME, HORIZON)
            ax.set_xticks([FIRST_FRAME, 120000, 240000, 360000, 480000, HORIZON])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}k"))
            ax.grid(alpha=0.2, linewidth=0.5)
            if metric in {"any_contact", "any_simultaneous_contact", "predator_boundary_fraction",
                          "prey_boundary_fraction"}:
                ax.set_ylim(-0.02, 1.02)
            if metric == "action_saturated_fraction":
                peak = max((value for r in records
                            for value in r["curves"].get((phase, metric), {}).values()
                            if math.isfinite(value)), default=0)
                ax.set_ylim(0, max(0.001, min(1, peak * 1.1)))
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=2))
            if metric == "any_simultaneous_contact":
                ax.text(0.02, 0.97, "≥2 predators contact the same prey", va="top",
                        transform=ax.transAxes, fontsize=7, color="0.35")
        handles = []
        for index, record in enumerate(records):
            run = record["run"]
            passed = run.get("validation_passed", False)
            status = record.get("managed_status", "missing/unverified")
            qualifier = "run checks passed" if passed else "run checks failed/incomplete"
            if record["issues"]:
                qualifier += "; plot issues"
            handles.append(Line2D([], [], color=colors(index % 10),
                                  linestyle="-" if passed else "--",
                                  label=f"Seed {run.get('seed', '?')}: {status}; {qualifier}"))
        title = f"{report.get('protocol_id', 'PCP confirmation')} | {runtime} | fixed 600k frames"
        fig.suptitle(title, fontsize=13, y=0.985)
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.95),
                   ncol=min(2, max(1, len(handles))), frameon=False, fontsize=8)
        checks = "passed" if report.get("checks_passed") else "FAILED / incomplete"
        observed = []
        for record in records:
            curve = record["curves"].get(("evaluation", "return_mean"), {})
            last = f"{max(curve) / 1000:g}k" if curve else "unavailable"
            observed.append(f"seed {record['run'].get('seed', '?')}: {last}")
        points = ("within-seed evaluation means" if name == "pcp_outcomes"
                  else "logged predator health summaries")
        footer = (
            f"Each line is one training seed; points are {points}. "
            "Descriptive fixed-budget evidence; no convergence claim.\n"
            f"Last observed evaluation — {', '.join(observed)}.\n"
            f"Report checks: {checks}; review: {report.get('review_status', 'unknown')}. "
            f"Report SHA256 {report_hash[:12]}; "
            f"source {str(report.get('source_sha256', 'unknown'))[:12]}."
        )
        if review is not None:
            footer += f"\nFull confirmation: {review['overall_verdict']}. {review['plot_note']}"
        fig.text(0.01, 0.015, footer, fontsize=7, va="bottom", color="0.3")
        fig.tight_layout(rect=(0, 0.075, 1, 0.94))
        for suffix in ("pdf", "png"):
            fig.savefig(out_dir / f"{name}.{suffix}", metadata={
                "Title": title, "Subject" if suffix == "pdf" else "Description": footer,
            })
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--review", type=Path, help="Final scientific review bound to this report.")
    parser.add_argument(
        "--suite", type=Path, help="Relocated suite; artifact hashes must still match."
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report_path, out_dir = args.report.resolve(), args.out_dir.resolve()
        report = _read_json(report_path)
        if report.get("schema_version") != 1 or not isinstance(report.get("runs", []), list):
            raise ValueError("Expected a schema_version=1 confirm_pcp.py report.")
        suite_value = args.suite or report.get("suite_dir")
        suite = Path(suite_value).resolve() if suite_value else None
        if suite and (out_dir == suite or out_dir.is_relative_to(suite)):
            raise ValueError("--out-dir must be outside the managed suite.")
        if report_path in [out_dir / name for name in OUTPUT_NAMES]:
            raise ValueError("Output would overwrite the input report.")
        report_hash = _sha256(report_path)
        review, review_binding = None, None
        if args.review:
            review_path = args.review.resolve()
            if review_path in [out_dir / name for name in OUTPUT_NAMES]:
                raise ValueError("Output would overwrite the input review.")
            review = _read_json(review_path)
            for key, expected in (
                ("confirmation_report_sha256", report_hash),
                ("source_sha256", report.get("source_sha256")),
                ("protocol_sha256", report.get("protocol_sha256")),
            ):
                if review.get(key) != expected:
                    raise ValueError(f"Scientific review {key} does not match this report.")
            if any(not isinstance(review.get(key), str) or not review[key].strip()
                   for key in ("overall_verdict", "plot_note")):
                raise ValueError("Scientific review requires overall_verdict and plot_note.")
            review_binding = {"path": str(review_path), "sha256": _sha256(review_path),
                              "overall_verdict": review["overall_verdict"]}
        runs = list(report.get("runs", []))
        if any(not isinstance(run, dict) for run in runs):
            raise ValueError("Every report run must be an object.")
        actual = {run.get("seed") for run in runs}
        runs.extend({"seed": seed, "validation_passed": False, "issues": []}
                    for seed in report.get("expected_seeds", []) if seed not in actual)
        if not runs:
            runs = [{"seed": None, "validation_passed": False,
                     "issues": report.get("issues", [])}]
        records = [_record(run, report, suite) for run in runs]
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(out_dir / "per_seed.csv", [_seed_row(record, report) for record in records])
        curve_rows = [
            {"seed": record["run"].get("seed"), "run_id": record["run"].get("run_id", ""),
             "phase": phase, "group": "adversary", "metric": metric, "frames": frame,
             "value": value if math.isfinite(value) else "", "finite": math.isfinite(value)}
            for record in records for (phase, metric), curve in sorted(record["curves"].items())
            for frame, value in sorted(curve.items())
        ]
        _write_csv(out_dir / "curves.csv", curve_rows, columns=[
            "seed", "run_id", "phase", "group", "metric", "frames", "value", "finite",
        ])
        _figures(records, report, out_dir, report_hash, review)
        provenance = {
            "schema_version": 1, "created_utc": datetime.now(UTC).isoformat(),
            "report": {"path": str(report_path), "sha256": report_hash},
            "scientific_review": review_binding,
            "protocol_id": report.get("protocol_id"),
            "protocol_sha256": report.get("protocol_sha256"),
            "training_source_sha256": report.get("source_sha256"),
            "report_checks_passed": report.get("checks_passed", False),
            "review_status": report.get("review_status"), "report_issues": report.get("issues", []),
            "suite": str(suite) if suite else None, "suite_override": args.suite is not None,
            "plotter": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__)),
                        "python": platform.python_version(), "matplotlib": matplotlib.__version__},
            "unit": "training seed; no episode pooling or cross-seed uncertainty estimate",
            "budget": {"first_frame": FIRST_FRAME, "maximum_frame": HORIZON,
                       "missing_observations": "gaps; no extrapolation or smoothing"},
            "runs": [{"seed": r["run"].get("seed"), "run_id": r["run"].get("run_id"),
                      "raw_curves_accepted": r["raw_curves_accepted"], "artifacts": r["artifacts"],
                      "issues": r["issues"]} for r in records],
            "outputs": {name: _sha256(out_dir / name) for name in OUTPUT_NAMES
                        if name != "provenance.json"},
        }
        (out_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"out_dir": str(out_dir), "seed_rows": len(records),
                          "report_checks_passed": report.get("checks_passed", False),
                          "plot_issues": sum(len(r["issues"]) for r in records)}, indent=2))
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
