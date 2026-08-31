"""Render one finished suite as a single readable markdown report.

The aggregate CSVs are the machine-readable record; this module is the human
entry point. It exists because a directory of eight CSVs and six PNGs does not
tell anyone what happened, which seeds are missing, or whether the numbers can
be trusted.

Everything here is derived from the artifacts the suite already wrote, so the
report can be regenerated at any time and never becomes a second, divergent
source of truth.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _number(value: Any, digits: int = 2) -> str:
    if value in (None, "", "None"):
        return "--"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "non-finite"
    return f"{numeric:,.{digits}f}"


def _integer(value: Any) -> str:
    if value in (None, "", "None"):
        return "--"
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return str(value)


def _interval(row: Mapping[str, Any], low: str, high: str, digits: int = 1) -> str:
    if not row.get(low) or not row.get(high):
        return "--"
    return f"[{_number(row[low], digits)}, {_number(row[high], digits)}]"


def _method(row: Mapping[str, Any]) -> str:
    return str(row.get("model", "?")).replace("comm_", "")


def _table(header: Sequence[str], rows: Sequence[Sequence[str]], align: str = "") -> list[str]:
    if not rows:
        return ["_No rows._", ""]
    alignment = align or ("l" + "r" * (len(header) - 1))
    separator = [
        {"l": "---", "r": "---:", "c": ":---:"}[kind] for kind in alignment[: len(header)]
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(separator) + "|"]
    lines.extend("| " + " | ".join(cells) + " |" for cells in rows)
    lines.append("")
    return lines


def _provenance(suite_dir: Path) -> dict[str, Any]:
    """Summarize the environment every completed run actually executed in."""

    commits: set[str] = set()
    dirty: set[bool] = set()
    devices: set[str] = set()
    gpus: set[str] = set()
    versions: dict[str, set[str]] = defaultdict(set)
    frames: set[int] = set()

    for run_dir in sorted(path for path in suite_dir.iterdir() if path.is_dir()):
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        git = metadata.get("git") or {}
        if git.get("commit_sha"):
            commits.add(str(git["commit_sha"])[:12])
        if "dirty" in git:
            dirty.add(bool(git["dirty"]))
        runtime = metadata.get("runtime") or {}
        if runtime.get("train_device"):
            devices.add(str(runtime["train_device"]))
        if runtime.get("cuda_device_name"):
            gpus.add(str(runtime["cuda_device_name"]))
        for package, version in (metadata.get("versions") or {}).items():
            if version:
                versions[package].add(str(version))
        if metadata.get("frames"):
            frames.add(int(metadata["frames"]))

    return {
        "commits": sorted(commits),
        "dirty": dirty,
        "devices": sorted(devices),
        "gpus": sorted(gpus),
        "versions": {name: sorted(values) for name, values in sorted(versions.items())},
        "frames": sorted(frames),
    }


def _inventory(per_run: Sequence[Mapping[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in per_run:
        counts[row.get("status") or "unknown"] += 1
    return dict(sorted(counts.items()))


def _main_rows(summaries: Sequence[Mapping[str, str]]) -> list[list[str]]:
    rows = []
    for row in sorted(summaries, key=_method):
        if row.get("ablation") not in (None, "", "main"):
            continue
        if not row.get("mean_final_return"):
            continue
        rows.append(
            [
                _method(row),
                str(row.get("n_seeds") or "0"),
                _number(row.get("mean_final_return"), 1),
                _interval(row, "ci95_low", "ci95_high"),
                _number(row.get("mean_normalized_auc"), 1),
                _number(row.get("mean_saliency_return_delta"), 1),
                _integer(row.get("actor_params")),
                _integer(row.get("comm_params")),
                _integer(row.get("mean_comm_realized_bits_per_step")),
            ]
        )
    return rows


def _saliency_rows(summaries: Sequence[Mapping[str, str]]) -> list[list[str]]:
    rows = []
    for row in sorted(summaries, key=_method):
        if row.get("ablation") not in (None, "", "main"):
            continue
        if not row.get("mean_saliency_return_delta"):
            continue
        rows.append(
            [
                _method(row),
                str(row.get("n_saliency_seeds") or "0"),
                _number(row.get("mean_saliency_return_with_comm"), 1),
                _number(row.get("mean_saliency_return_without_comm"), 1),
                _number(row.get("mean_saliency_return_delta"), 1),
                _interval(row, "saliency_return_delta_ci95_low", "saliency_return_delta_ci95_high"),
                _number(row.get("mean_saliency_return_delta_fraction"), 3),
                _number(row.get("mean_saliency_action_shift_mean"), 3),
            ]
        )
    return rows


def _per_seed_rows(per_run: Sequence[Mapping[str, str]]) -> tuple[list[str], list[list[str]]]:
    by_method: dict[str, dict[int, str]] = defaultdict(dict)
    for row in per_run:
        if row.get("status") != "completed" or not row.get("mean_final_return"):
            continue
        if row.get("ablation") not in (None, "", "main"):
            continue
        by_method[_method(row)][int(row["seed"])] = _number(row["mean_final_return"], 1)
    seeds = sorted({seed for values in by_method.values() for seed in values})
    header = ["Method", *(f"seed {seed}" for seed in seeds)]
    rows = [
        [method, *(by_method[method].get(seed, "--") for seed in seeds)]
        for method in sorted(by_method)
    ]
    return header, rows


def _ablation_sections(summaries: Sequence[Mapping[str, str]]) -> list[str]:
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in summaries:
        ablation = row.get("ablation")
        if ablation not in (None, "", "main"):
            grouped[str(ablation)].append(row)

    def sort_key(row: Mapping[str, str]) -> tuple[str, int, float | str]:
        value = row.get("ablation_value")
        try:
            return (_method(row), 0, float(value))
        except (TypeError, ValueError):
            return (_method(row), 1, str(value))

    lines: list[str] = []
    for ablation, rows in sorted(grouped.items()):
        lines.append(f"### {ablation.replace('_', ' ')}")
        lines.append("")
        lines.extend(
            _table(
                ["Method", "Value", "Seeds", "Final return", "95% CI", "AUC", "Saliency"],
                [
                    [
                        _method(row),
                        str(row.get("ablation_value") or "--"),
                        str(row.get("n_seeds") or "0"),
                        _number(row.get("mean_final_return"), 1),
                        _interval(row, "ci95_low", "ci95_high"),
                        _number(row.get("mean_normalized_auc"), 1),
                        _number(row.get("mean_saliency_return_delta"), 1),
                    ]
                    for row in sorted(rows, key=sort_key)
                    if row.get("mean_final_return")
                ],
            )
        )
    return lines


def _comparison_rows(comparisons: Sequence[Mapping[str, str]]) -> list[list[str]]:
    rows = []
    for row in comparisons:
        if row.get("metric") != "mean_final_return":
            continue
        if row.get("ablation") not in (None, "", "main"):
            continue
        if not row.get("mean_paired_difference"):
            continue
        rows.append(
            [
                _method(row),
                str(row.get("n_pairs") or "0"),
                _number(row.get("mean_paired_difference"), 1),
                _interval(row, "difference_ci95_low", "difference_ci95_high"),
                _number(row.get("paired_effect_size_cohens_dz"), 2),
            ]
        )
    return rows


def render_report(suite_dir: Path, results_dir: Path) -> str:
    """Build the markdown report body for one analyzed suite."""

    suite_dir = suite_dir.resolve()
    results_dir = results_dir.resolve()
    prefix = suite_dir.name

    per_run = _read_csv(results_dir / f"{prefix}_per_run.csv")
    summaries = _read_csv(results_dir / f"{prefix}_summary.csv")
    failed = _read_csv(results_dir / f"{prefix}_failed_runs.csv")
    comparisons = _read_csv(results_dir / f"{prefix}_paired_comparisons.csv")
    provenance = _provenance(suite_dir)
    inventory = _inventory(per_run)

    lines: list[str] = [
        f"# {prefix}",
        "",
        f"_Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} from "
        f"`{suite_dir.name}/`._",
        "",
        "## Run inventory",
        "",
    ]
    total = sum(inventory.values())
    lines.extend(
        _table(
            ["Status", "Rows", "Share"],
            [
                [status, str(count), f"{100 * count / total:.0f}%" if total else "--"]
                for status, count in inventory.items()
            ],
        )
    )

    incomplete = total - inventory.get("completed", 0)
    if incomplete:
        lines.append(
            f"> **{incomplete} of {total} rows are not completed.** Aggregates below "
            "use only completed rows; treat them as provisional."
        )
        lines.append("")

    lines.extend(["## Provenance", ""])
    version_cells = [
        [name, ", ".join(values)] for name, values in provenance["versions"].items()
    ]
    dirty = provenance["dirty"]
    lines.extend(
        _table(
            ["Property", "Value"],
            [
                ["git commit", ", ".join(provenance["commits"]) or "--"],
                [
                    "working tree",
                    # Never collapse a partially dirty suite to a neutral word:
                    # "some runs dirty" is the fact a reader must act on.
                    "dirty (patch saved per run)"
                    if dirty == {True}
                    else "clean"
                    if dirty == {False}
                    else "mixed - some runs dirty, some clean"
                    if dirty
                    else "--",
                ],
                ["train device", ", ".join(provenance["devices"]) or "--"],
                ["gpu", ", ".join(provenance["gpus"]) or "n/a"],
                [
                    "frames per run",
                    ", ".join(f"{value:,}" for value in provenance["frames"]) or "--",
                ],
                *version_cells,
            ],
            align="ll",
        )
    )
    # Mixing GPU models is the same hazard as mixing CPU and CUDA: kernel
    # selection and reduction order differ between architectures, so a study
    # split across V100 and H100 nodes is not a clean comparison.
    if (
        len(provenance["commits"]) > 1
        or len(provenance["devices"]) > 1
        or len(provenance["gpus"]) > 1
    ):
        mixed = []
        if len(provenance["commits"]) > 1:
            mixed.append(f"commits ({', '.join(provenance['commits'])})")
        if len(provenance["devices"]) > 1:
            mixed.append(f"devices ({', '.join(provenance['devices'])})")
        if len(provenance["gpus"]) > 1:
            mixed.append(f"GPU models ({', '.join(provenance['gpus'])})")
        lines.append(
            f"> **Rows in this suite did not all execute under one environment.** "
            f"Mixed {'; '.join(mixed)}. Numerics differ across commits, devices, and "
            "GPU architectures, so cross-row comparison is only valid within a "
            "single one. Pin the GPU model when submitting, e.g. "
            "`sbatch --gres=gpu:h100:1 ...`."
        )
        lines.append("")

    lines.extend(
        [
            "## Main comparison",
            "",
            "Final return is the mean of the final 10% of evaluation points. AUC is "
            "the trapezoidal return-vs-frames integral divided by its frame span. "
            "Intervals are fixed-RNG 95% bootstrap over seeds.",
            "",
        ]
    )
    lines.extend(
        _table(
            [
                "Method",
                "Seeds",
                "Final return",
                "95% CI",
                "AUC",
                "Saliency",
                "Actor params",
                "Comm params",
                "Bits/step",
            ],
            _main_rows(summaries),
        )
    )

    header, rows = _per_seed_rows(per_run)
    if rows:
        lines.extend(["### Per-seed final return", ""])
        lines.extend(_table(header, rows))

    comparison_rows = _comparison_rows(comparisons)
    if comparison_rows:
        lines.extend(
            [
                "### Matched-seed difference vs Identity",
                "",
                "Positive means the method beat the no-communication control on the "
                "same seeds. `dz` is Cohen's d for paired differences.",
                "",
            ]
        )
        lines.extend(
            _table(
                ["Method", "Pairs", "Mean difference", "95% CI", "dz"],
                comparison_rows,
            )
        )

    saliency_rows = _saliency_rows(summaries)
    lines.extend(["## Communication saliency", ""])
    if saliency_rows:
        lines.extend(
            [
                "Return lost when every message is suppressed on the frozen trained "
                "policy, evaluated paired from the same environment seed. Zero is the "
                "correct value for the MLP and Identity controls. A large action shift "
                "with a small return delta means the channel changes behaviour without "
                "improving the shared objective.",
                "",
            ]
        )
        lines.extend(
            _table(
                [
                    "Method",
                    "Seeds",
                    "With comm",
                    "Severed",
                    "Saliency",
                    "95% CI",
                    "Relative",
                    "Action shift",
                ],
                saliency_rows,
            )
        )
    else:
        lines.extend(
            [
                "_Not measured. Run `scripts/saliency.py` over this suite, then "
                "re-run `scripts/analyze.py` and `scripts/report.py`._",
                "",
            ]
        )

    ablation_lines = _ablation_sections(summaries)
    if ablation_lines:
        lines.extend(["## Ablations", ""])
        lines.extend(ablation_lines)

    unavailable = [row for row in failed if row.get("status") != "completed"]
    lines.extend(["## Failed and unavailable rows", ""])
    if unavailable:
        lines.extend(
            _table(
                ["Run", "Status", "Error", "Message"],
                [
                    [
                        row.get("run_id", "?"),
                        row.get("status", "?"),
                        row.get("error_type") or "--",
                        (row.get("error_message") or "--")[:80],
                    ]
                    for row in unavailable[:50]
                ],
                align="llll",
            )
        )
        if len(unavailable) > 50:
            lines.append(f"_{len(unavailable) - 50} further rows omitted._")
            lines.append("")
    else:
        lines.extend(["None. Every planned row completed.", ""])

    plots = sorted(results_dir.glob("*.png"))
    if plots:
        lines.extend(["## Figures", ""])
        for plot in plots:
            title = plot.stem.replace("_", " ").capitalize()
            lines.append(f"### {title}")
            lines.append("")
            lines.append(f"![{title}]({plot.name})")
            lines.append("")

    lines.extend(
        [
            "## Artifacts",
            "",
            *(f"- `{path.name}`" for path in sorted(results_dir.glob("*.csv"))),
            "",
        ]
    )
    return "\n".join(lines)


def write_report(suite_dir: Path, results_dir: Path) -> Path:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "REPORT.md"
    path.write_text(render_report(Path(suite_dir), results_dir), encoding="utf-8")
    return path
