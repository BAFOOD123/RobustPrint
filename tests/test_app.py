"""Smoke test of the Streamlit dashboard (headless)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


@pytest.mark.slow
def test_dashboard_renders_and_optimises() -> None:
    at = AppTest.from_file(str(APP), default_timeout=240).run()
    assert not at.exception, at.exception
    assert any("SYNTHETIC" in info.value for info in at.info)
    assert len(at.metric) == 3
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert len(at.dataframe) >= 1
