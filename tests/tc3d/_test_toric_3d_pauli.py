import pytest
import numpy as np
from panqec.bpauli import bsf_wt
from panqec.codes import Toric3DCode
import panqec.bsparse as bsparse
from scipy.sparse import csr_matrix


@pytest.fixture
def code():
    return Toric3DCode(9, 10, 11)


class TestToric3DPauli:

    def test_x_block_z_block_to_bsf_and_back(self, code):
        L_x, L_y, L_z = code.size

        # Deterministic example of X and Z blocks in 4D array.
        np.random.seed(0)
        x_block = np.random.randint(0, 2, size=3*L_x*L_y*L_z)
        z_block = np.random.randint(0, 2, size=3*L_x*L_y*L_z)

        # Ensure non-trivial.
        assert np.any(z_block == 1)
        assert np.any(x_block == 1)
        assert np.any(z_block != x_block)

        # Manually inject the X and Z blocks into hidden attributes.
        operator = Toric3DPauli(code)

        # Test that xs and zs are eithe np array or csr matrices
        assert isinstance(operator._xs, np.ndarray) or isinstance(operator._xs, csr_matrix)
        assert isinstance(operator._zs, np.ndarray) or isinstance(operator._zs, csr_matrix)

        if isinstance(operator._xs, csr_matrix):
            x_block = bsparse.from_array(x_block)
            z_block = bsparse.from_array(x_block)

        operator._xs = x_block
        operator._zs = z_block

        # Convert to binary simplectic form.
        bsf = operator.to_bsf()
        if isinstance(operator._xs, np.ndarray):
            assert bsf.shape[0] == 2*3*L_x*L_y*L_z
        else:
            assert bsf.shape[1] == 2*3*L_x*L_y*L_z

        # Convert back into X and Z block arrays.
        new_operator = Toric3DPauli(code, bsf=bsf)

        if isinstance(operator._xs, np.ndarray):
            assert np.all(new_operator._xs == x_block)
            assert np.all(new_operator._zs == z_block)
        else:
            assert bsparse.equal(new_operator._xs, x_block)
            assert bsparse.equal(new_operator._zs, z_block)

    def test_apply_operator_on_site(self, code):
        operator = Toric3DPauli(code)
        operator.site('X', (0, 0, 1))

        assert operator.operator((0, 0, 1)) == 'X'
        assert bsf_wt(operator.to_bsf()) == 1

    def test_apply_operators_on_vertex(self, code):
        vertex_operator = Toric3DPauli(code)
        vertex_operator.vertex('X', (2, 2, 2))

        # Each vertex has 6 edges.
        assert bsf_wt(vertex_operator.to_bsf()) == 6

        # Same operator applied site by site.
        operator = dict()
        operator[(2, 3, 2)] = 'X'
        operator[(3, 2, 2)] = 'X'
        operator[(1, 2, 2)] = 'X'
        operator[(2, 1, 2)] = 'X'
        operator[(2, 2, 1)] = 'X'
        operator[(2, 2, 3)] = 'X'

        assert bsf_wt(operator.to_bsf()) == 6

        if isinstance(operator.to_bsf(), np.ndarray):
            assert np.all(vertex_operator.to_bsf() == operator.to_bsf())
        else:
            assert bsparse.equal(vertex_operator.to_bsf(), operator.to_bsf())

        assert vertex_operator == operator

    def test_apply_face_operators_on_x_normal_face(self, code):
        face_operator = Toric3DPauli(code)
        face_operator.face('X', (0, 1, 1))

        # Each face has 4 edges.
        assert bsf_wt(face_operator.to_bsf()) == 4

    def test_apply_face_operators_on_y_normal_face(self, code):
        face_operator = Toric3DPauli(code)
        face_operator.face('X', (1, 0, 1))

        # Each face has 4 edges.
        assert bsf_wt(face_operator.to_bsf()) == 4

    def test_apply_face_operators_on_z_normal_face(self, code):
        face_operator = Toric3DPauli(code)
        face_operator.face('X', (1, 1, 0))

        # Each face has 4 edges.
        assert bsf_wt(face_operator.to_bsf()) == 4

    def test_make_a_2x2_loop(self, code):
        loop = Toric3DPauli(code)
        loop.face('X', (0, 1, 1))
        loop.face('X', (0, 3, 1))
        loop.face('X', (0, 1, 3))
        loop.face('X', (0, 3, 3))
        assert bsf_wt(loop.to_bsf()) == 8

    def test_make_a_tetris_T_loop(self, code):
        """Product of faces like this should have weight 10.

        +-Z-+-Z-+-Z-+
        Z   |   |   Z
        +-Z-+---+-Z-+
            Z   Z
            +-Z-+

        """
        loop = Toric3DPauli(code)
        loop.face('Z', (3, 2, 1))
        loop.face('Z', (3, 2, 3))
        loop.face('Z', (1, 2, 3))
        loop.face('Z', (5, 2, 3))
        assert bsf_wt(loop.to_bsf()) == 10

    def test_faces_sharing_same_vertex(self, code):
        r"""This thing should have weight six.

             .---Y---.            Coordinate axes:
            /       / \                      y
           Y       /   Y                    /
          /       /     \                  /
         .-------o       .         z <----o
          \       \     /                  \
           Y       \   Y                    \
            \       \ /                      x
             .---Y---.

        """
        loop = Toric3DPauli(code)
        loop.face('Y', (1, 0, 1))
        loop.face('Y', (0, 1, 1))
        loop.face('Y', (1, 1, 2))

        assert bsf_wt(loop.to_bsf()) == 6
