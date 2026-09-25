import math
from pathlib import Path

from matplotlib.figure import Figure
from omegaconf import OmegaConf
import pandas as pd
import pytest

from commstudy.analysis.plotting import aggregate, plot_metrics, read_metrics, seed_curves
from commstudy.experiments.bookkeeping import atomic_write_json
from commstudy.experiments.config import load_config


def test_equal_seed_weight_completed_only_and_six_figures(tmp_path, monkeypatch):
    config = load_config(Path(__file__).resolve().parents[1] / "configs/pcp.yaml")
    config.update(seeds=[1100, 1101, 1102], output_dir=str(tmp_path))
    (tmp_path / "config.yaml").write_text(OmegaConf.to_yaml(config))
    records = []
    for seed, frames, values in [
        (1100, 8, [0, 10]),
        (1101, 8, [30]),
        (1102, 8, [999]),
        (1100, 16, [20]),
    ]:
        for sample, value in enumerate(values):
            records.append(
                dict(
                    method="identity",
                    seed=seed,
                    frames=frames,
                    phase="evaluation",
                    group="",
                    metric="return_mean",
                    value=value,
                    sample=sample,
                )
            )
    pd.DataFrame(records).to_csv(tmp_path / "metrics.csv", index=False)
    for seed, state in [(1100, "completed"), (1101, "completed"), (1102, "failed")]:
        atomic_write_json(tmp_path / "identity" / f"seed_{seed}" / "status.json", {"state": state})
    labels = []
    save = Figure.savefig

    def capture(figure, *args, **kwargs):
        labels.extend(
            label for axis in figure.axes for label in axis.get_legend_handles_labels()[1]
        )
        return save(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    paths = plot_metrics(tmp_path, ["evaluation/return_mean"])
    assert len(paths) == len(list((tmp_path / "plots").glob("*.png"))) == 6
    assert "Identity (2/3 seeds)" in labels
    assert "Broadcast (0/3 seeds)" in labels
    assert "Seed 1102 (failed)" in labels
    plotted = pd.read_csv(tmp_path / "plots/evaluation__return_mean__aggregate.csv")
    assert list(plotted.frames) == [8]  # no extrapolation to the unmatched frame 16
    assert plotted["mean"].iloc[0] == 17.5
    assert plotted["std"].iloc[0] == pytest.approx(math.sqrt(312.5))
    assert plotted.seeds.iloc[0] == 2
    data = read_metrics(tmp_path)
    assert set(data.name) == {"evaluation/return_mean"}
    curves = seed_curves(data, "evaluation/return_mean", "identity")
    assert curves.loc[8, 1100] == 5
    assert aggregate(curves, []).empty
