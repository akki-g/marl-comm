"""Exercise the installed checkpoint writer without collecting or training."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping

import numpy as np
import pytest
import torch

from commstudy.experiments import build_experiment, load_experiment_spec
from commstudy.utils.rng import RNGState, preserve_rng_state


def _assert_equal(before, after):
    if isinstance(before, torch.Tensor):
        assert before.dtype == after.dtype and before.shape == after.shape
        assert torch.equal(before, after)
    elif isinstance(before, np.ndarray):
        assert before.dtype == after.dtype
        np.testing.assert_array_equal(before, after)
    elif isinstance(before, Mapping):
        assert before.keys() == after.keys()
        for key in before:
            _assert_equal(before[key], after[key])
    elif isinstance(before, (list, tuple)):
        assert type(before) is type(after) and len(before) == len(after)
        for left, right in zip(before, after, strict=True):
            _assert_equal(left, right)
    else:
        assert before == after


@pytest.mark.parametrize("keep", [2, None], ids=["bounded_retention", "retain_all"])
def test_real_pcp_checkpoint_writer_duplicate_final_save_retention(
    config_root, tmp_path, keep,
):
    """Periodic 600k and checkpoint_at_end both save the same filename.

    Frame counters below are fixture labels, not a simulated training result.
    The real Experiment and its state_dict/writer are used without run().
    """
    with preserve_rng_state():
        spec = load_experiment_spec(config_root, [
            "task=vmas_predator_capture_prey", "model=pcp_comm_identity",
            "critic_model=pcp_critic", "experiment.max_n_frames=6000",
            "experiment.on_policy_n_envs_per_worker=1", "experiment.evaluation=false",
            "experiment.loggers=[]", "experiment.create_json=false",
            f"experiment.keep_checkpoints_num={'null' if keep is None else keep}",
            f"experiment.save_folder={tmp_path}",
        ])
        experiment = build_experiment(spec)
        try:
            checkpoint_dir = experiment.folder_name / "checkpoints"
            checkpoint_480k = checkpoint_dir / "checkpoint_480000.pt"
            saved_480k_hash = None
            for frame in (120000, 240000, 360000, 480000, 600000, 600000):
                experiment.total_frames = frame
                before = copy.deepcopy(experiment.state_dict())
                before_rng = RNGState.capture()
                experiment._save_experiment()
                _assert_equal(before, experiment.state_dict())
                _assert_equal(vars(before_rng), vars(RNGState.capture()))
                loaded = torch.load(
                    checkpoint_dir / f"checkpoint_{frame}.pt", weights_only=False,
                )
                _assert_equal(before, loaded)
                assert loaded["state"]["n_iters_performed"] == 0
                assert "loss_adversary" in loaded and "loss_agent" in loaded
                if frame == 480000:
                    saved_480k_hash = hashlib.sha256(checkpoint_480k.read_bytes()).hexdigest()
            names = {path.name for path in checkpoint_dir.glob("checkpoint_*.pt")}
            if keep is None:
                assert names == {
                    f"checkpoint_{frame}.pt"
                    for frame in (120000, 240000, 360000, 480000, 600000)
                }
                assert hashlib.sha256(checkpoint_480k.read_bytes()).hexdigest() == saved_480k_hash
                assert len(experiment._checkpointed_files) == 6
            else:
                assert names == {"checkpoint_600000.pt"}
                assert not checkpoint_480k.exists()
                assert len(experiment._checkpointed_files) == 2
            assert list(experiment._checkpointed_files)[-2:] == [
                checkpoint_dir / "checkpoint_600000.pt",
                checkpoint_dir / "checkpoint_600000.pt",
            ]
        finally:
            experiment.close()
