"""Shared fixtures. Net-dependent tests skip unless LEELA_SAE_CHECKPOINT is set, e.g.

    LEELA_SAE_CHECKPOINT=lczerolens/BT3-768x15x24h-swa-2790000 uv run pytest

LEELA_SAE_RESID_MODULE optionally overrides the residual site (default: module.encoder7/ln2).
"""

import os

import pytest

CHECKPOINT = os.environ.get("LEELA_SAE_CHECKPOINT")
RESID_MODULE = os.environ.get("LEELA_SAE_RESID_MODULE", "module.encoder7/ln2")

requires_net = pytest.mark.skipif(not CHECKPOINT, reason="set LEELA_SAE_CHECKPOINT to run tests on the real net")


@pytest.fixture(scope="session")
def model():
    from src import hook
    return hook.load_model(CHECKPOINT)


@pytest.fixture(scope="session")
def resid_module():
    return RESID_MODULE
