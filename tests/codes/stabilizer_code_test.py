import pytest
from typing import Tuple, List
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.special import comb
from scipy.sparse import find
from tqdm import tqdm
from itertools import combinations
from panqec.codes import StabilizerCode
from panqec.bsparse import is_sparse, vstack, to_array
from panqec.bpauli import bs_prod, brank, bvector_to_pauli_string


class StabilizerCodeTest(metaclass=ABCMeta):

    @pytest.fixture
    @abstractmethod
    def code(self) -> StabilizerCode:
        pass

    def test_n_equals_len_qubit_index(self, code):
        assert code.n == len(code.qubit_coordinates)

    def test_len_vertex_index_equals_number_of_vertex_stabilizers(self, code):
        if 'vertex' in code.stabilizer_types and code.is_css:
            n_vertices = len(code.type_index('vertex'))
            assert n_vertices == code.Hz.shape[0]

    def test_len_face_index_equals_number_of_face_stabilizers(self, code):
        if 'face' in code.stabilizer_types and code.is_css:
            n_faces = len(code.type_index('face'))
            assert n_faces == code.Hx.shape[0]

    def test_qubit_stabilizer_indices_no_overlap(self, code):
        qubits = set(code.qubit_index.keys())
        stabilizers = set(code.stabilizer_index.keys())
        assert qubits.isdisjoint(stabilizers)

    def test_all_stabilizers_commute(self, code):
        commutators = bs_prod(code.stabilizer_matrix, code.stabilizer_matrix)
        print_non_commuting(
            code, commutators, code.stabilizer_matrix, code.stabilizer_matrix,
            'stabilizer', 'stabilizer'
        )

        # There should be no non-commuting pairs of stabilizers.
        assert np.all(commutators == 0)

    def test_logicals_same_size(self, code):
        assert len(code.logicals_x) == len(code.logicals_z)

    def test_logical_operators_anticommute_pairwise(self, code):
        k = code.k
        assert np.all(bs_prod(code.logicals_x, code.logicals_x) == 0)
        assert np.all(bs_prod(code.logicals_z, code.logicals_z) == 0)
        commutators = bs_prod(code.logicals_x, code.logicals_z)
        assert np.all(commutators == np.eye(k)), (
            f'Not pairwise anticommuting {commutators}'
        )

    def test_logical_operators_commute_with_stabilizers(self, code):
        x_commutators = bs_prod(code.logicals_x, code.stabilizer_matrix)
        print_non_commuting(
            code, x_commutators, code.logicals_x, code.stabilizer_matrix,
            'logicalX', 'stabilizer'
        )
        z_commutators = bs_prod(code.logicals_z, code.stabilizer_matrix)
        print_non_commuting(
            code, z_commutators, code.logicals_z, code.stabilizer_matrix,
            'logicalZ', 'stabilizer'
        )
        assert np.all(to_array(x_commutators) == 0), (
            'logicalX not commuting with stabilizers'
        )
        assert np.all(to_array(z_commutators) == 0), (
            'logicalZ not commuting with stabilizers'
        )

    def test_logical_operators_are_independent_by_rank(self, code):
        n, k = code.n, code.k
        matrix = code.stabilizer_matrix

        matrix_with_logicals = vstack([
            matrix,
            code.logicals_x,
            code.logicals_z,
        ])

        rank_with_logicals = brank(matrix_with_logicals)
        assert rank_with_logicals == n + k

    def test_n_indepdent_stabilizers_equals_n_minus_k(self, code):
        n, k = code.n, code.k
        matrix = code.stabilizer_matrix

        # Number of independent stabilizer generators.
        rank = brank(matrix)

        assert rank <= matrix.shape[0]
        assert rank == n - k

    def test_number_of_logicals_is_k(self, code):
        k = code.k
        assert code.logicals_x.shape[0] == k
        assert code.logicals_z.shape[0] == k


class StabilizerCodeTestWithCoordinates(StabilizerCodeTest, metaclass=ABCMeta):

    @property
    @abstractmethod
    def size(self) -> Tuple[int, int, int]:
        """Plane size x"""

    @property
    @abstractmethod
    def code_class(self):
        """The code Class to be tested"""

    @property
    @abstractmethod
    def expected_plane_edges_xy(self) -> List[Tuple[int, int]]:
        """Expected xy coordinates of xy plane edges in unrotated lattice."""

    @property
    @abstractmethod
    def expected_plane_faces_xy(self) -> List[Tuple[int, int]]:
        """Expected xy coordinates of xy plane faces in unrotated lattice."""

    @property
    @abstractmethod
    def expected_vertical_faces_xy(self) -> List[Tuple[int, int]]:
        """Expected xy coordinates of xy plane faces in unrotated lattice."""

    @property
    @abstractmethod
    def expected_plane_vertices_xy(self) -> List[Tuple[int, int]]:
        """Expected xy coordinates of xy plane vertices in unrot lattice."""

    @property
    @abstractmethod
    def expected_plane_z(self) -> List[int]:
        """Expected z coordinates of horizontal planes."""

    @property
    @abstractmethod
    def expected_vertical_z(self) -> List[int]:
        """Expected z coordinates of vertical edges and faces."""

    @pytest.fixture
    def code(self):
        return self.code_class(*self.size)

    def test_qubit_indices(self, code):
        locations = set(code.qubit_coordinates)
        expected_locations = set([
            (x, y, z)
            for x, y in self.expected_plane_edges_xy
            for z in self.expected_plane_z
        ] + [
            (x, y, z)
            for x, y in self.expected_plane_vertices_xy
            for z in self.expected_vertical_z
        ])
        assert locations == expected_locations

    def test_vertex_indices(self, code):
        locations = set(code.type_index('vertex').keys())
        expected_locations = set([
            (x, y, z)
            for x, y in self.expected_plane_vertices_xy
            for z in self.expected_plane_z
        ])
        assert locations == expected_locations

    def test_face_indices(self, code):
        locations = set(code.type_index('face').keys())
        expected_locations = set([
            (x, y, z)
            for x, y in self.expected_vertical_faces_xy
            for z in self.expected_vertical_z
        ] + [
            (x, y, z)
            for x, y in self.expected_plane_faces_xy
            for z in self.expected_plane_z
        ])
        assert locations == expected_locations

    @pytest.mark.skip(reason='brute force')
    def test_find_lowest_weight_Z_only_logical_by_brute_force(self, code):
        n, k = code.n, code.k
        matrix = code.stabilizer_matrix

        coords = {v: k for v, k in enumerate(code.qubit_coordinates)}

        min_weight = 4
        max_weight = 4
        for weight in range(min_weight, max_weight + 1):
            n_comb = comb(n, weight, exact=True)
            for sites in tqdm(combinations(range(n), weight), total=n_comb):
                logical = np.zeros(2*n, dtype=np.uint)
                for site in sites:
                    x, y, z = coords[site]
                    deform = False
                    if z % 2 == 1 and (x + y) % 4 == 2:
                        deform = True

                    if deform:
                        logical[site] = 1  # X operator on deformed
                    else:
                        logical[n + site] = 1  # Z operator on undeformed

                codespace = np.all(bs_prod(matrix, logical) == 0)
                if codespace:
                    matrix_with_logical = np.concatenate([
                        to_array(matrix),
                        [logical]
                    ])
                    if brank(matrix_with_logical) == n - k + 1:
                        print('Found Z-only logical')
                        print([coords[site] for site in sites])
                        print(operator_spec(code, logical))
                        return


def operator_spec(code, bsf):
    """Get representation of BSF as list of (pauli, (x, y, z)) entries.

    Useful for debugging and reading BSF in human-readable format.
    """
    operator_spec = []
    if len(bsf.shape) != 1:
        bsf_1d = to_array(bsf)[0]
    else:
        bsf_1d = to_array(bsf)
    pauli_string = bvector_to_pauli_string(bsf_1d)
    for index, pauli in enumerate(pauli_string):
        if pauli != 'I':
            matches = [
                xyz
                for i, xyz in enumerate(code.qubit_coordinates)
                if i == index
            ]
            location = matches[0]
            operator_spec.append((pauli, location))
    return operator_spec


def print_non_commuting(
    code: StabilizerCode, commutators: np.ndarray,
    operators_1: np.ndarray, operators_2: np.ndarray,
    name_1: str, name_2: str,
    max_print: int = 5
):
    if is_sparse(commutators):
        non_commuting = set([
            (i, j)
            for i, j in np.array(find(commutators))[:2].T
            if i <= j
        ])
    else:
        non_commuting = set([
            (i, j)
            for i, j in np.array(np.where(commutators)).T
            if i <= j
        ])

    # Print the first few non-commuting stabilizers if any found.
    if non_commuting:
        print(
            f'{len(non_commuting)} pairs of non-commuting '
            f'{name_1} and {name_2}:'
        )
        for i_print, (i, j) in enumerate(non_commuting):
            try:
                loc_1 = code.stabilizer_coordinates[i]
            except IndexError:
                loc_1 = ''
            try:
                loc_2 = code.stabilizer_coordinates[j]
            except IndexError:
                loc_2 = ''
            print(
                f'{name_1} {i} {loc_1} and {name_2} {j} {loc_2} anticommuting'
            )
            operator_spec_1 = operator_spec(code, operators_1[i])
            operator_spec_2 = operator_spec(code, operators_2[j])
            overlap = [
                (op_1, op_2, site_1)
                for op_1, site_1 in operator_spec_1
                for op_2, site_2 in operator_spec_2
                if site_1 == site_2
            ]
            print('Overlap:', overlap)
            print(f'{name_1} {i}:', operator_spec_1)
            print(f'{name_2} {j}:', operator_spec_2)
            if i_print == max_print:
                n_remaining = len(non_commuting) - 1 - i_print
                if n_remaining > 0:
                    print(f'... {n_remaining} more pairs')
                break
