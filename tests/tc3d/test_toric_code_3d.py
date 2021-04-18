import numpy as np
import pytest
from qecsim.paulitools import bsf_wt
from bn3d.tc3d import ToricCode3D


class TestToricCode3D:

    @pytest.fixture()
    def code(self):
        return ToricCode3D(5)

    def test_get_vertex_Z_stabilizers(self, code):
        vertex_stabilizers = code.get_vertex_Z_stabilizers()

        # There should be least some vertex stabilizers.
        assert len(vertex_stabilizers) > 0
        assert vertex_stabilizers.dtype == np.uint

        # All Z stabilizers should be weight 6.
        assert all(
            bsf_wt(stabilizer) == 6 for stabilizer in vertex_stabilizers
        )
