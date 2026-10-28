"""Shared fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from robustprint.data.synthetic import generate_dataset
from robustprint.physics.simulator import ISS_NOMINAL


@pytest.fixture(scope="session")
def small_dataset() -> pd.DataFrame:
    """A small, fixed-seed synthetic campaign (30 batches x 4 settings)."""
    frame, _ = generate_dataset(ISS_NOMINAL, 30, 4, np.random.default_rng(123))
    return frame
