import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xsent.config import Config  # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    c = Config()
    c.data_dir = tmp_path / "data"
    c.out_dir = tmp_path / "out"
    c.apify_token = "test-token"
    c.tavily_api_key = "tvly-test"
    c.lookback_days = 3
    c.ensure_dirs()
    return c
