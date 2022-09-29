import pytest
import numpy as np
from panqec.bpauli import bsf_wt
from panqec.codes import Toric3DCode


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

        # Construc the bsf.
        bsf = np.hstack([x_block, z_block])
        operator = code.from_bsf(bsf)

        # Convert to binary simplectic form.
        new_bsf = code.to_bsf(operator)
        assert np.all(new_bsf == bsf)

    def test_apply_operator_on_site(self, code):
        operator = {
            (0, 0, 1): 'X'
        }
        assert bsf_wt(code.to_bsf(operator)) == 1

    def test_apply_operators_on_vertex(self, code):
        location = (2, 2, 2)
        assert code.stabilizer_type(location) == 'vertex'
        vertex_operator = code.get_stabilizer(location)

        # Each vertex has 6 edges.
        assert bsf_wt(code.to_bsf(vertex_operator)) == 6

        # Same operator applied site by site.
        operator = dict()
        operator[(2, 3, 2)] = 'Z'
        operator[(3, 2, 2)] = 'Z'
        operator[(1, 2, 2)] = 'Z'
        operator[(2, 1, 2)] = 'Z'
        operator[(2, 2, 1)] = 'Z'
        operator[(2, 2, 3)] = 'Z'

        assert bsf_wt(code.to_bsf(operator)) == 6

        from panqec.utils import simple_print
        simple_print(code.to_bsf(vertex_operator))
        simple_print(code.to_bsf(operator))
        assert np.all(
            code.to_bsf(vertex_operator) == code.to_bsf(operator)
        )

        assert vertex_operator == operator

    @pytest.mark.parametrize('location', [(0, 1, 1), (1, 0, 1), (1, 1, 0)])
    def test_apply_face_operators_on_face(self, code, location):
        assert code.stabilizer_type(location) == 'face'
        face_operator = code.get_stabilizer(location)

        # Each face has 4 edges.
        assert bsf_wt(code.to_bsf(face_operator)) == 4

    def test_make_a_2x2_loop(self, code):
        faces = [(0, 1, 1), (0, 3, 1), (0, 1, 3), (0, 3, 3)]
        assert all(code.stabilizer_type(face) == 'face' for face in faces)
        loop = code.from_bsf(sum(
            code.to_bsf(code.get_stabilizer(face))
            for face in faces
        ) % 2)
        assert bsf_wt(code.to_bsf(loop)) == 8
        assert loop == {
            (0, 1, 0): 'X', (0, 3, 0): 'X',
            (0, 4, 1): 'X', (0, 4, 3): 'X',
            (0, 3, 4): 'X', (0, 1, 4): 'X',
            (0, 0, 3): 'X', (0, 0, 1): 'X',
        }

    def test_make_a_tetris_T_loop(self, code):
        """Product of faces like this should have weight 10.

        +-Z-+-Z-+-Z-+
        Z   |   |   Z
        +-Z-+---+-Z-+
            Z   Z
            +-Z-+

        """
        faces = [
            (3, 2, 1), (3, 2, 3), (1, 2, 3), (5, 2, 3),
        ]
        loop = code.from_bsf(sum(
            code.to_bsf(code.get_stabilizer(face))
            for face in faces
        ) % 2)
        assert bsf_wt(code.to_bsf(loop)) == 10

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
        faces = [(1, 0, 1), (0, 1, 1), (1, 1, 2)]
        loop = code.from_bsf(sum(
            code.to_bsf(code.get_stabilizer(face))
            for face in faces
        ) % 2)

        assert bsf_wt(code.to_bsf(loop)) == 6
