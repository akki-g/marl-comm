from pathlib import Path
import os

for name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[name] = "1"
os.environ["TQDM_DISABLE"] = "1"

import pytest  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)


@pytest.fixture(scope="session")
def config_root():
    return Path(__file__).resolve().parents[1] / "configs"
