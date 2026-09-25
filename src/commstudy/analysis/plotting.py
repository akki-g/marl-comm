"""Plot equal-weight seed curves at recorded environment frames."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from commstudy.experiments.bookkeeping import read_json
from commstudy.experiments.config import METHODS, load_config


def read_metrics(root):
    data = pd.read_csv(Path(root) / "metrics.csv", keep_default_na=False)
    data["name"] = (
        data["phase"]
        + "/"
        + data["group"].where(data["group"].eq(""), data["group"] + "/")
        + data["metric"]
    )
    return data


def seed_curves(data, metric, method):
    rows = data[(data["name"] == metric) & (data["method"] == method)]
    # Episode/sample observations first become one observation per seed/frame.
    return rows.groupby(["seed", "frames"], sort=True)["value"].mean().unstack("seed")


def aggregate(curves, completed):
    columns = [seed for seed in completed if seed in curves.columns]
    if not columns:
        return pd.DataFrame(columns=["mean", "std", "seeds"])
    # Intersection only: never interpolate or change seed weight along a curve.
    aligned = curves[columns].dropna()
    return pd.DataFrame(
        {
            "mean": aligned.mean(axis=1),
            "std": aligned.std(axis=1, ddof=1).fillna(0),
            "seeds": len(columns),
        }
    )


def plot_metrics(root, metrics):
    root = Path(root)
    config, data = load_config(root / "config.yaml"), read_metrics(root)
    missing = set(metrics) - set(data["name"])
    if missing:
        raise ValueError(f"Unknown metrics: {', '.join(sorted(missing))}; use --list-metrics")
    output = root / "plots"
    output.mkdir(exist_ok=True)
    paths = []
    for metric in dict.fromkeys(metrics):
        safe_name = metric.replace("/", "__")
        if not safe_name or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
            for c in safe_name
        ):
            raise ValueError(f"Metric cannot be used as a filename: {metric}")
        figure, axis = plt.subplots(figsize=(8, 5))
        rows = []
        for method in METHODS:
            states = {
                seed: read_json(root / method / f"seed_{seed}" / "status.json", {}).get(
                    "state", "pending"
                )
                for seed in config["seeds"]
            }
            completed = [seed for seed, state in states.items() if state == "completed"]
            curves = seed_curves(data, metric, method)
            result = aggregate(curves, completed)
            usable = len([seed for seed in completed if seed in curves.columns])
            label = f"{method.title()} ({usable}/{len(states)} seeds)"
            (line,) = axis.plot(result.index, result["mean"], label=label)
            if not result.empty:
                axis.fill_between(
                    result.index,
                    result["mean"] - result["std"],
                    result["mean"] + result["std"],
                    color=line.get_color(),
                    alpha=0.18,
                )
                rows.append(
                    result.reset_index().assign(
                        method=method, metric=metric, expected_seeds=len(states)
                    )
                )
            seed_figure, seed_axis = plt.subplots(figsize=(8, 5))
            for seed, state in states.items():
                if seed in curves:
                    series = curves[seed].dropna()
                    seed_axis.plot(
                        series.index,
                        series,
                        label=f"Seed {seed} ({state})",
                        linestyle="-" if state == "completed" else "--",
                    )
                else:
                    seed_axis.plot([], [], label=f"Seed {seed} ({state}; no data)")
            seed_axis.set(
                title=f"{method.title()}: {metric}", xlabel="Environment frames", ylabel=metric
            )
            seed_axis.legend(fontsize="small")
            seed_axis.grid(alpha=0.2)
            seed_figure.tight_layout()
            path = output / f"{safe_name}__{method}.png"
            seed_figure.savefig(path, dpi=160)
            plt.close(seed_figure)
            paths.append(path)
        axis.set(title=f"{metric} — seed mean ± 1 SD", xlabel="Environment frames", ylabel=metric)
        axis.legend(fontsize="small")
        axis.grid(alpha=0.2)
        figure.tight_layout()
        path = output / f"{safe_name}__comparison.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(path)
        aggregated = (
            pd.concat(rows, ignore_index=True)
            if rows
            else pd.DataFrame(
                columns=["frames", "mean", "std", "seeds", "method", "metric", "expected_seeds"]
            )
        )
        aggregated.to_csv(output / f"{safe_name}__aggregate.csv", index=False)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--metrics", nargs="+")
    parser.add_argument("--list-metrics", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.list_metrics:
            print("\n".join(sorted(read_metrics(args.run)["name"].unique())))
            return 0
        if not args.metrics:
            parser.error("Specify --metrics or --list-metrics")
        for path in plot_metrics(args.run, args.metrics):
            print(path)
        return 0
    except (ValueError, OSError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
