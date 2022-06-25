from importlib import reload
import pytest
import os
import panqec
from panqec.config import PANQEC_DIR, PANQEC_DARK_THEME


def test_output_dir_exists():
    assert os.path.exists(PANQEC_DIR)


def test_error_raised_if_panqec_dir_does_not_exist(monkeypatch):
    monkeypatch.setenv('PANQEC_DIR', '/fake/path/here/')
    with pytest.raises(FileNotFoundError):
        reload(panqec.config)


def test_dark_theme_known():
    assert type(PANQEC_DARK_THEME) is bool
