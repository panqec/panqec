import os
from bn3d.config import BN3D_DIR, BN3D_DARK_THEME


def test_output_dir_exists():
    assert os.path.exists(BN3D_DIR)


def test_dark_theme_known():
    assert type(BN3D_DARK_THEME) is bool
