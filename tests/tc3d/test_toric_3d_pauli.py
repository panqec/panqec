import pytest
import numpy as np
from qecsim.paulitools import bsf_wt
from bn3d.tc3d import ToricCode3D, Toric3DPauli


@pytest.fixture
def code():
    return ToricCode3D(3, 4, 5)


class TestToric3DPauli:

    def test_x_block_z_block_to_bsf_and_back(self, code):
        _, L_x, L_y, L_z = code.shape

        # Deterministic example of X and Z blocks in 4D array.
        np.random.seed(0)
        x_block = np.random.randint(0, 2, size=(3, L_x, L_y, L_z))
        z_block = np.random.randint(0, 2, size=(3, L_x, L_y, L_z))

        # Ensure non-trivial.
        assert np.any(z_block == 1)
        assert np.any(x_block == 1)
        assert np.any(z_block != x_block)

        # Manually inject the X and Z blocks into hidden attributes.
        operator = Toric3DPauli(code)
        operator._xs = x_block
        operator._zs = z_block

        # Convert to binary simplectic form.
        bsf = operator.to_bsf()
        assert bsf.shape[0] == 2*3*L_x*L_y*L_z

        # Convert back into X and Z block arrays.
        new_operator = Toric3DPauli(code, bsf=bsf)
        assert np.all(new_operator._xs == x_block)
        assert np.all(new_operator._zs == z_block)

    def test_apply_operator_on_site(self, code):
        operator = Toric3DPauli(code)
        operator.site('X', (1, 2, 3, 4))

        assert operator.operator((1, 2, 3, 4)) == 'X'
        assert bsf_wt(operator.to_bsf()) == 1

    def test_apply_operators_on_vertex(self, code):
        vertex_operator = Toric3DPauli(code)
        vertex_operator.vertex('X', (2, 3, 4))

        # Each vertex has 6 edges.
        assert bsf_wt(vertex_operator.to_bsf()) == 6

        # Same operator applied site by site.
        operator = Toric3DPauli(code)
        operator.site('X', (0, 2, 3, 4))
        operator.site('X', (0, 1, 3, 4))
        operator.site('X', (1, 2, 3, 4))
        operator.site('X', (1, 2, 2, 4))
        operator.site('X', (2, 2, 3, 4))
        operator.site('X', (2, 2, 3, 3))

        assert bsf_wt(operator.to_bsf()) == 6

        assert np.all(vertex_operator.to_bsf() == operator.to_bsf())
        assert vertex_operator == operator

    def test_apply_operators_on_vertex_periodic_boundary(self, code):
        origin_vertex = Toric3DPauli(code)
        origin_vertex.vertex('Z', (0, 0, 0))

        assert bsf_wt(origin_vertex.to_bsf()) == 6

        # Operator on the far vertex (secretly the same because periodic).
        far_vertex = Toric3DPauli(code)
        L_x, L_y, L_z = code.size
        far_vertex .vertex('Z', (L_x, L_y, L_z))

        assert origin_vertex == far_vertex
