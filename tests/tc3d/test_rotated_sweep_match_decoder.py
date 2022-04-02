import pytest
from itertools import combinations
import numpy as np
from qecsim.paulitools import bsf_wt
from panqec.bpauli import bcommute
from panqec.codes import (
    RotatedPlanar3DCode
)
from panqec.decoders import RotatedSweepMatchDecoder, RotatedSweepDecoder3D


@pytest.mark.skip(reason='sparse')
class TestRotatedSweepMatchDecoder:

    @pytest.fixture
    def code(self):
        return RotatedPlanar3DCode(4, 4, 4)

    @pytest.fixture
    def decoder(self):
        return RotatedSweepMatchDecoder()

    def test_decoder_has_required_attributes(self, decoder):
        assert decoder.label is not None
        assert decoder.decode is not None

    def test_decode_trivial_syndrome(self, decoder, code):
        syndrome = np.zeros(shape=len(code.stabilizer_matrix), dtype=np.uint)
        correction = decoder.decode(code, syndrome)
        assert correction.shape[0] == 2*code.n
        assert np.all(bcommute(code.stabilizer_matrix, correction) == 0)
        assert issubclass(correction.dtype.type, np.integer)

    @pytest.mark.parametrize(
        'pauli, location',
        [
            ('X', (3, 3, 1)),
            ('Z', (6, 4, 8)),
            ('Y', (7, 9, 5)),
        ]
    )
    def test_decode_single_error(self, decoder, code, pauli, location):
        error = dict()
        assert location in code.qubit_coordinates
        error[location] = pauli
        assert bsf_wt(code.to_bsf(error)) == 1

        # Measure the syndrome and ensure non-triviality.
        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        correction = decoder.decode(code, syndrome)
        total_error = (code.to_bsf(error) + correction) % 2
        assert np.all(bcommute(code.stabilizer_matrix, total_error) == 0)

    @pytest.mark.parametrize('sweep_direction, diffs', [
        [(+1, 0, +1), [(+1, +1, +1), (+1, -1, +1), (+2, 0, 0)]],
        [(+1, 0, -1), [(+1, +1, -1), (+1, -1, -1), (+2, 0, 0)]],
        [(0, +1, +1), [(+1, +1, +1), (-1, +1, +1), (0, +2, 0)]],
        [(0, +1, -1), [(+1, +1, -1), (-1, +1, -1), (0, +2, 0)]],
        [(-1, 0, +1), [(-1, -1, +1), (-1, +1, +1), (-2, 0, 0)]],
        [(-1, 0, -1), [(-1, -1, -1), (-1, +1, -1), (-2, 0, 0)]],
        [(0, -1, +1), [(-1, -1, +1), (+1, -1, +1), (0, -2, 0)]],
        [(0, -1, -1), [(-1, -1, -1), (+1, -1, -1), (0, -2, 0)]],
    ])
    def test_get_sweep_faces(self, sweep_direction, diffs, decoder, code):
        vertex = (0, 0, 0)
        expected_x_face, expected_y_face, expected_z_face = [
            tuple(np.array(vertex) + np.array(diff))
            for diff in diffs
        ]
        x_face, y_face, z_face = decoder._sweeper.get_sweep_faces(
            vertex, sweep_direction, code
        )
        assert x_face == expected_x_face
        assert y_face == expected_y_face
        assert z_face == expected_z_face

    @pytest.mark.parametrize('sweep_direction, diffs', [
        [(+1, 0, +1), [(+1, -1, 0), (+1, +1, 0), (0, 0, +1)]],
        [(+1, 0, -1), [(+1, -1, 0), (+1, +1, 0), (0, 0, -1)]],
        [(0, +1, +1), [(-1, +1, 0), (+1, +1, 0), (0, 0, +1)]],
        [(0, +1, -1), [(-1, +1, 0), (+1, +1, 0), (0, 0, -1)]],
        [(-1, 0, +1), [(-1, +1, 0), (-1, -1, 0), (0, 0, +1)]],
        [(-1, 0, -1), [(-1, +1, 0), (-1, -1, 0), (0, 0, -1)]],
        [(0, -1, +1), [(+1, -1, 0), (-1, -1, 0), (0, 0, +1)]],
        [(0, -1, -1), [(+1, -1, 0), (-1, -1, 0), (0, 0, -1)]],
    ])
    def test_get_sweep_edges(self, sweep_direction, diffs, decoder, code):
        vertex = (0, 0, 0)
        expected_x_edge, expected_y_edge, expected_z_edge = [
            tuple(np.array(vertex) + np.array(diff))
            for diff in diffs
        ]
        x_edge, y_edge, z_edge = decoder._sweeper.get_sweep_edges(
            vertex, sweep_direction, code
        )
        assert x_edge == expected_x_edge
        assert y_edge == expected_y_edge
        assert z_edge == expected_z_edge

    @pytest.mark.parametrize(
        'paulis_locations',
        [
            [('X', (3, 3, 1)), ('X', (7, 9, 5)), ('X', (6, 4, 8))],
            [('Z', (3, 3, 1)), ('Z', (7, 9, 5)), ('Z', (6, 4, 8))],
            [('Y', (9, 5, 1)), ('Y', (2, 12, 4)), ('Y', (6, 8, 4))],
            [('X', (1, 1, 1)), ('X', (1, 3, 1))],
            [('X', (1, 1, 1)), ('X', (3, 1, 1))],
            [('X', (2, 0, 2)), ('X', (2, 0, 4))],
            [('Z', (1, 1, 1)), ('Z', (1, 3, 1))],
            [('Z', (1, 1, 1)), ('Z', (3, 1, 1))],
            [('Z', (2, 0, 2)), ('Z', (2, 0, 4))],
            [('Y', (1, 1, 1)), ('Y', (1, 3, 1))],
            [('Y', (1, 1, 1)), ('Y', (3, 1, 1))],
            [('Y', (2, 0, 2)), ('Y', (2, 0, 4))],
            [('Z', (1, 1, 1))],
        ],
        ids=[
            'X_bulk', 'Z_bulk', 'Y_bulk',
            'X_boundary_x', 'X_boundary_y', 'X_boundary_z',
            'Z_boundary_x', 'Z_boundary_y', 'Z_boundary_z',
            'Y_boundary_x', 'Y_boundary_y', 'Y_boundary_z',
            'Z_corner'
        ]
    )
    def test_decode_many_errors(self, decoder, code, paulis_locations):
        error = dict()
        for pauli, location in paulis_locations:
            assert location in code.qubit_coordinates
            error[location] = pauli
        assert bsf_wt(code.to_bsf(error)) == len(paulis_locations)

        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        correction = decoder.decode(code, syndrome)
        total_error = (code.to_bsf(error) + correction) % 2
        assert np.all(bcommute(code.stabilizer_matrix, total_error) == 0)

    def test_undecodable_error(self, decoder, code):
        locations = [
            (x, y, z) for x, y, z in code.qubit_coordinates
        ]
        assert len(locations) > 0
        error = dict()
        for location in locations:
            assert location in code.qubit_coordinates
            error[location] = 'Z'
        assert bsf_wt(code.to_bsf(error)) == len(locations)

        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        correction = decoder.decode(code, syndrome)
        total_error = (code.to_bsf(error) + correction) % 2
        assert np.any(total_error)

    def test_decode_many_codes_and_errors_with_same_decoder(self, decoder):

        codes_sites = [
            (RotatedPlanar3DCode(3, 3, 3), (7, 9, 3)),
            (RotatedPlanar3DCode(4, 4, 4), (3, 5, 7)),
            (RotatedPlanar3DCode(5, 5, 5), (1, 3, 5)),
        ]

        for code, site in codes_sites:
            error = dict()
            assert site in code.qubit_coordinates
            error[site] = 'Z'
            syndrome = code.measure_syndrome(error)
            correction = decoder.decode(code, syndrome)
            total_error = (code.to_bsf(error) + correction) % 2
            assert np.all(bcommute(code.stabilizer_matrix, total_error) == 0)


@pytest.mark.skip(reason='sparse')
class TestSweepMatch1x1x1:
    """Test cases found to be failing on the GUI."""

    @pytest.fixture
    def code(self):
        return RotatedPlanar3DCode(1, 1, 1)

    @pytest.fixture
    def decoder(self):
        return RotatedSweepMatchDecoder()

    @pytest.mark.parametrize('locations', [
        [('Z', (1, 5, 1)), ('Z', (1, 5, 3))],
        [('Z', (1, 1, 1)), ('Z', (3, 3, 1)), ('Z', (5, 5, 1))],
        [('Z', (1, 1, 3)), ('Z', (3, 3, 3)), ('Z', (5, 5, 3))],
        [('Z', (3, 5, 3)), ('Z', (5, 3, 3)), ('Z', (5, 1, 3))],
    ], ids=[
        'z_vertical',
        'up_left_horizontal_bottom',
        'up_left_horizontal_top',
        'down_right_horizontal'
    ])
    def test_errors_spanning_boundaries(self, code, decoder, locations):
        error = dict()
        for pauli, location in locations:
            error[location] = pauli
        assert bsf_wt(code.to_bsf(error)) == len(locations)

        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        correction = decoder.decode(code, syndrome)
        total_error = (code.to_bsf(error) + correction) % 2
        assert not np.all(bcommute(code.stabilizer_matrix, total_error) == 0), (
            'Total error not in codespace'
        )


@pytest.mark.skip(reason='sparse')
class TestSweepMatch2x2x2:
    """Test cases found to be failing on the GUI."""

    @pytest.fixture
    def code(self):
        return RotatedPlanar3DCode(2, 2, 2)

    @pytest.fixture
    def decoder(self):
        return RotatedSweepMatchDecoder()

    @pytest.mark.parametrize('locations', [
        [
            ('Z', (1, 1, 5)), ('Z', (3, 3, 5)), ('Z', (5, 5, 5)),
            ('Z', (7, 5, 5)), ('Z', (9, 3, 5)),
        ],
        [
            ('Z', (1, 1, 3)), ('Z', (1, 3, 5)),
        ],
        [
            ('Z', (1, 1, 5)), ('Z', (1, 3, 5)),
        ],
        [
            ('Z', (1, 3, 1)), ('Z', (3, 1, 5)),
        ],
        [
            ('Z', (1, 3, 1)), ('Z', (4, 2, 4)),
        ],
        [
            ('Z', (1, 3, 3)), ('Z', (3, 1, 5)),
        ],
    ], ids=[
        'arthurs_example',
        'weight_2_Z_error_1',
        'weight_2_Z_error_2',
        'weight_2_Z_error_3',
        'weight_2_Z_error_4',
        'weight_2_Z_error_5',
    ])
    def test_gui_examples(self, code, decoder, locations):
        error = dict()
        for pauli, location in locations:
            assert location in code.qubit_coordinates
            error[location] = pauli
        assert bsf_wt(code.to_bsf(error)) == len(locations)

        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        correction = decoder.decode(code, syndrome)
        total_error = (code.to_bsf(error) + correction) % 2
        assert np.all(bcommute(code.stabilizer_matrix, total_error) == 0), (
            'Total error not in codespace'
        )

        assert np.all(bcommute(code.logicals_x, total_error) == 0), (
            'Total error anticommutes with logical X'
        )
        assert np.all(bcommute(code.logicals_z, total_error) == 0), (
            'Total error anticommutes with logical Z'
        )

    @pytest.mark.parametrize('locations', [
        [('Z', (1, 9, 1)), ('Z', (1, 9, 3)), ('Z', (1, 9, 5))],
        [
            ('Z', (1, 1, 1)), ('Z', (3, 3, 1)), ('Z', (5, 5, 1)),
            ('Z', (7, 7, 1)), ('Z', (9, 9, 1))
        ],
        [
            ('Z', (1, 1, 5)), ('Z', (3, 3, 5)), ('Z', (5, 5, 5)),
            ('Z', (7, 7, 5)), ('Z', (9, 9, 5))
        ],
        [
            ('Z', (3, 9, 5)), ('Z', (5, 7, 5)), ('Z', (7, 5, 5)),
            ('Z', (9, 3, 5)), ('Z', (9, 1, 5))
        ],
    ], ids=[
        'z_vertical',
        'up_left_horizontal_bottom',
        'up_left_horizontal_top',
        'down_right_horizontal',
    ])
    def test_errors_spanning_boundaries(self, code, decoder, locations):
        error = dict()
        for pauli, location in locations:
            assert location in code.qubit_coordinates
            error[location] = pauli
        assert bsf_wt(code.to_bsf(error)) == len(locations)

        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        correction = decoder.decode(code, syndrome)
        total_error = (code.to_bsf(error) + correction) % 2
        assert not np.all(bcommute(code.stabilizer_matrix, total_error) == 0), (
            'Total error in codespace'
        )


@pytest.mark.skip(reason='sparse')
class TestSweepCorners:
    """Test 1-qubit errors on corners fully correctable."""

    @pytest.fixture
    def code(self):
        return RotatedPlanar3DCode(2, 2, 2)

    @pytest.fixture
    def decoder(self):
        return RotatedSweepMatchDecoder()

    @pytest.mark.parametrize('location', [
        (1, 3, 5),
        (3, 5, 5),
        (5, 7, 5),
        (7, 9, 5),
        (9, 9, 5)
    ])
    def test_sweep_errors_on_extreme_layer(self, code, decoder, location):
        error = dict()
        assert location in code.qubit_coordinates
        error[location] = 'Z'
        assert bsf_wt(code.to_bsf(error)) == 1

        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        correction = decoder.decode(code, syndrome)
        total_error = (code.to_bsf(error) + correction) % 2
        assert np.all(bcommute(code.stabilizer_matrix, total_error) == 0), (
            'Total error not in codespace'
        )

        assert np.all(bcommute(code.logicals_x, total_error) == 0), (
            'Total error anticommutes with logical X'
        )
        assert np.all(bcommute(code.logicals_z, total_error) == 0), (
            'Total error anticommutes with logical Z'
        )

    @pytest.mark.parametrize('pauli', ['X', 'Y', 'Z'])
    def test_all_1_qubit_errors_correctable(self, code, decoder, pauli):
        uncorrectable_locations = []
        for location in code.qubit_coordinates:
            error = dict()
            error[location] = pauli
            assert bsf_wt(code.to_bsf(error)) == 1

            syndrome = code.measure_syndrome(error)
            assert np.any(syndrome != 0)

            correction = decoder.decode(code, syndrome)
            total_error = (code.to_bsf(error) + correction) % 2
            assert np.all(bcommute(code.stabilizer_matrix, total_error) == 0)

            correctable = True
            if np.any(bcommute(code.logicals_x, total_error) != 0):
                correctable = False
            if np.all(bcommute(code.logicals_z, total_error) != 0):
                correctable = False
            if not correctable:
                uncorrectable_locations.append(location)

        assert len(uncorrectable_locations) == 0, (
            f'Found {len(uncorrectable_locations)} uncorrectable weight-1 '
            f'{pauli} errors'
        )

    @pytest.mark.slow
    def test_all_2_qubit_errors_correctable(self, code, decoder):
        pauli = 'Z'
        weight = 2
        error_locations = combinations(list(code.qubit_index), weight)
        uncorrectable_error_locations = []
        for locations in error_locations:
            error = dict()
            for location in locations:
                error[location] = pauli
            assert bsf_wt(code.to_bsf(error)) == len(locations)

            syndrome = code.measure_syndrome(error)
            assert np.any(syndrome != 0)

            correction = decoder.decode(code, syndrome)
            total_error = (code.to_bsf(error) + correction) % 2
            assert np.all(bcommute(code.stabilizer_matrix, total_error) == 0)

            correctable = True
            if np.any(bcommute(code.logicals_x, total_error) != 0):
                correctable = False
            if np.all(bcommute(code.logicals_z, total_error) != 0):
                correctable = False
            if not correctable:
                uncorrectable_error_locations.append(locations)

        assert len(uncorrectable_error_locations) == 0, (
            f'Found {len(uncorrectable_error_locations)} uncorrectable '
            f'weight-2 Z errors'
        )


@pytest.mark.skip(reason='sparse')
class TestRotatedSweepDecoder3D:

    @pytest.fixture
    def code(self):
        return RotatedPlanar3DCode(2, 2, 2)

    @pytest.fixture
    def decoder(self):
        return RotatedSweepDecoder3D()

    @pytest.mark.parametrize(
        'vertex,sweep_direction,sweep_faces,sweep_edges',
        [
            (
                (4, 6, 5),
                (1, 0, -1),
                [(5, 7, 4), (5, 5, 4), (6, 6, 5)],
                [(5, 5, 5), (5, 7, 5), (4, 6, 4)],
            )
        ]
    )
    def test_sweep_faces_edges_top_boundary(
        self, code, decoder, vertex, sweep_direction, sweep_faces, sweep_edges
    ):
        vertex = (4, 6, 5)
        sweep_direction = (1, 0, -1)
        assert vertex in code.vertex_index
        x_face, y_face, z_face = decoder.get_sweep_faces(
            vertex, sweep_direction, code
        )
        x_edge, y_edge, z_edge = decoder.get_sweep_edges(
            vertex, sweep_direction, code
        )
        assert [x_face, y_face, z_face] == sweep_faces
        assert [x_edge, y_edge, z_edge] == sweep_edges

    def test_adjacency_sweep_faces_edges(self, code, decoder):
        touched_edges = []
        touched_faces = []
        sweep_directions = [
            (1, 0, 1), (1, 0, -1),
            (0, 1, 1), (0, 1, -1),
            (-1, 0, 1), (-1, 0, -1),
            (0, -1, 1), (0, -1, -1),
        ]
        for sweep_direction in sweep_directions:
            for vertex in code.vertex_index:
                x_face, y_face, z_face = decoder.get_sweep_faces(
                    vertex, sweep_direction, code
                )
                x_edge, y_edge, z_edge = decoder.get_sweep_edges(
                    vertex, sweep_direction, code
                )

                faces_valid = tuple(
                    face in code.face_index
                    for face in [x_face, y_face, z_face]
                )
                edges_valid = tuple(
                    edge in code.qubit_coordinates
                    for edge in [x_edge, y_edge, z_edge]
                )
                if all(faces_valid) and all(edges_valid):
                    x_face_bsf = code.stabilizer_matrix[code.face_index[x_face]]
                    y_face_bsf = code.stabilizer_matrix[code.face_index[y_face]]
                    z_face_bsf = code.stabilizer_matrix[code.face_index[z_face]]

                    error = dict()
                    error[x_edge] = 'Z'
                    x_edge_bsf = code.to_bsf(error)
                    assert np.all(bcommute(x_edge_bsf, x_face_bsf) == 0)
                    assert np.any(bcommute(x_edge_bsf, y_face_bsf) == 1)
                    assert np.any(bcommute(x_edge_bsf, z_face_bsf) == 1)

                    error = dict()
                    error[y_edge] = 'Z'
                    y_edge_bsf = code.to_bsf(error)
                    assert np.all(bcommute(y_edge_bsf, y_face_bsf) == 0)
                    assert np.any(bcommute(y_edge_bsf, x_face_bsf) == 1)
                    assert np.any(bcommute(y_edge_bsf, z_face_bsf) == 1)

                    error = dict()
                    error[z_edge] = 'Z'
                    z_edge_bsf = code.to_bsf(error)
                    assert np.any(bcommute(z_edge_bsf, z_face_bsf) == 0)
                    assert np.any(bcommute(z_edge_bsf, x_face_bsf) == 1)
                    assert np.any(bcommute(z_edge_bsf, y_face_bsf) == 1)

                    touched_edges.append(x_edge)
                    touched_edges.append(y_edge)
                    touched_edges.append(z_edge)

                    touched_faces.append(x_face)
                    touched_faces.append(y_face)
                    touched_faces.append(z_face)

        assert set(code.qubit_coordinates) == set(touched_edges)
        assert set(code.face_index.keys()) == set(touched_faces)

    def test_flip_edge(self, code, decoder):
        n_faces = len(code.face_index)
        for edge in code.qubit_coordinates:
            signs = {face: 0 for face in code.face_index}
            decoder.flip_edge(edge, signs, code)
            sign_flip_syndrome = np.zeros(n_faces, dtype=int)
            for face, sign in signs.items():
                if sign:
                    sign_flip_syndrome[code.face_index[face]] = 1

            error = dict()
            error[edge] = 'Z'
            error_bsf = code.to_bsf(error)
            pauli_syndrome = bcommute(code.stabilizer_matrix[:n_faces], error_bsf)

            assert np.all(pauli_syndrome == sign_flip_syndrome)
