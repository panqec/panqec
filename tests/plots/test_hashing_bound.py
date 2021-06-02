import numpy as np
from bn3d.plots._hashing_bound import project_triangle


def test_project_triangle():
    assert np.allclose(project_triangle([1/3, 1/3, 1/3]), [0, 0])
