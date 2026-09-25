"""Small programmatic inputs for core integration tests."""

from dataclasses import replace

from commstudy.experiments.config import load_config, make_spec


def example_spec(
    config_root,
    *,
    task="pcp",
    method="identity",
    seed=0,
    experiment=None,
    task_params=None,
    channel=None,
):
    config = load_config(config_root / f"{task}.yaml")
    config["seeds"] = [seed]
    spec = make_spec(config, method, seed)
    if task_params:
        spec.task_config["params"].update(task_params)
    if channel:
        actor = spec.model_config["groups"]["adversary"] if task == "pcp" else spec.model_config
        actor["params"]["comm_kwargs"]["channel"] = channel
    return replace(spec, experiment={**spec.experiment, **(experiment or {})})
