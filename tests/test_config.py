import os
from bn3d.config import BN3D_DIR


def test_output_dir_exists():
    assert os.path.exists(BN3D_DIR)
