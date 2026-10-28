"""Renders every tab of the Streamlit dashboard headlessly (skipped if streamlit isn't installed)."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"


def test_dashboard_renders_without_errors(monkeypatch):
    monkeypatch.setenv("SENTINELML_API", "http://127.0.0.1:9")  # nothing listens: offline paths must not crash
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert [t.label for t in at.tabs] == ["Overview", "Live service", "What-if analysis", "Monitoring"]
