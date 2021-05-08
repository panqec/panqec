from importlib import reload
import pytest
import os
import bn3d
from bn3d.config import BN3D_DIR, BN3D_DARK_THEME


def test_output_dir_exists():
    assert os.path.exists(BN3D_DIR)


def test_error_raised_if_bn3d_dir_does_not_exist(monkeypatch):
    monkeypatch.setenv('BN3D_DIR', '/fake/path/here/')
    with pytest.raises(FileNotFoundError):
        reload(bn3d.config)


def test_dark_theme_known():
    assert type(BN3D_DARK_THEME) is bool
