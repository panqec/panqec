import pytest
from itertools import combinations
import numpy as np
from panqec.bpauli import bs_prod, bsf_wt
from panqec.codes import (
    RotatedPlanar3DCode
)
from panqec.decoders import RotatedSweepMatchDecoder, RotatedSweepDecoder3D
from panqec.error_models import PauliErrorModel
from tests.decoders.decoder_test import DecoderTest


class TestRotatedSweepMatchDecoder(DecoderTest):

    @pytest.fixture
    def code(self):
        return RotatedPlanar3DCode(4, 4, 4)

    @pytest.fixture
    def decoder(self, code, error_model):
        error_rate = 0.5
        return RotatedSweepMatchDecoder(code, error_model, error_rate,
                                        max_rounds=4)

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
    def test_get_sweep_faces(self, sweep_direction, diffs, decoder):
        vertex = (0, 0, 0)
        expected_x_face, expected_y_face, expected_z_face = [
            tuple(np.array(vertex) + np.array(diff))
            for diff in diffs
        ]
        x_face, y_face, z_face = decoder.sweeper.get_sweep_faces(
            vertex, sweep_direction
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
    def test_get_sweep_edges(self, sweep_direction, diffs, decoder):
        vertex = (0, 0, 0)
        expected_x_edge, expected_y_edge, expected_z_edge = [
            tuple(np.array(vertex) + np.array(diff))
            for diff in diffs
        ]
        x_edge, y_edge, z_edge = decoder.sweeper.get_sweep_edges(
            vertex, sweep_direction
        )
        assert x_edge == expected_x_edge
        assert y_edge == expected_y_edge
        assert z_edge == expected_z_edge

    @pytest.mark.parametrize(
        'paulis_locations',
        [
            [('X', (3, 3, 1)), ('X', (5, 1, 5)), ('X', (6, 4, 6))],
            [('Z', (3, 3, 1)), ('Z', (5, 1, 5)), ('Z', (6, 4, 6))],
            [('Y', (1, 5, 1)), ('Y', (2, 4, 4)), ('Y', (6, 4, 4))],
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
        error = code.to_bsf({
            location: pauli
            for pauli, location in paulis_locations
        })
        assert bsf_wt(error) == len(paulis_locations)

        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        correction = decoder.decode(syndrome)
        total_error = (error + correction) % 2
        assert np.all(code.measure_syndrome(total_error) == 0)

    def test_undecodable_error(self, decoder, code):
        locations = [
            (x, y, z) for x, y, z in code.qubit_coordinates
        ]
        assert len(locations) > 0
        error = code.to_bsf({
            location: 'Z'
            for location in locations
        })
        assert bsf_wt(error) == len(locations)

        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        correction = decoder.decode(syndrome)
        total_error = (error + correction) % 2
        assert np.any(total_error)

    def test_decode_many_codes_and_errors_with_same_decoder(self):

        codes_sites = [
            (RotatedPlanar3DCode(3, 3, 3), (3, 3, 3)),
            (RotatedPlanar3DCode(4, 4, 4), (5, 5, 5)),
            (RotatedPlanar3DCode(5, 5, 5), (3, 3, 3)),
        ]

        error_model = PauliErrorModel(1/3, 1/3, 1/3)
        error_rate = 0.5

        for code, site in codes_sites:
            decoder = RotatedSweepMatchDecoder(code, error_model, error_rate,
                                               max_rounds=4)
            error = code.to_bsf({site: 'Z'})
            syndrome = code.measure_syndrome(error)
            correction = decoder.decode(syndrome)
            total_error = (error + correction) % 2
            assert np.all(code.measure_syndrome(total_error) == 0)


class TestSweepMatch3x3x3(DecoderTest):
    """Test cases found to be failing on the GUI."""

    @pytest.fixture
    def code(self):
        return RotatedPlanar3DCode(3, 3, 3)

    @pytest.fixture
    def decoder(self, code, error_model):
        error_rate = 0.5
        return RotatedSweepMatchDecoder(code, error_model, error_rate)

    @pytest.mark.parametrize('locations', [
        [('Z', (1, 1, 1)), ('Z', (3, 3, 1)), ('Z', (5, 5, 1))],
        [('Z', (1, 1, 3)), ('Z', (3, 3, 3)), ('Z', (5, 5, 3))],
        [('Z', (3, 5, 3)), ('Z', (5, 3, 3)), ('Z', (5, 1, 3))],
    ], ids=[
        'up_left_horizontal_bottom',
        'up_left_horizontal_top',
        'down_right_horizontal'
    ])
    def test_errors_spanning_boundaries_fail(self, code, decoder, locations):
        error = code.to_bsf({
            location: pauli
            for pauli, location in locations
        })
        assert bsf_wt(error) == len(locations)

        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        correction = decoder.decode(syndrome)
        total_error = (error + correction) % 2
        assert not np.all(
            code.measure_syndrome(total_error) == 0
        ), 'Total error in codespace when it should not be'


class TestSweepMatch4x4x3(DecoderTest):
    """Test cases found to be failing on the GUI."""

    @pytest.fixture
    def code(self):
        return RotatedPlanar3DCode(5, 5, 5)

    @pytest.fixture
    def decoder(self, code, error_model):
        error_rate = 0.5
        return RotatedSweepMatchDecoder(code, error_model, error_rate,
                                        max_rounds=4)

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
        error = code.to_bsf({
            location: pauli
            for pauli, location in locations
        })
        assert bsf_wt(error) == len(locations)

        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        correction = decoder.decode(syndrome)
        total_error = (error + correction) % 2
        assert np.all(code.measure_syndrome(total_error) == 0), (
            'Total error not in codespace'
        )

        assert np.all(bs_prod(code.logicals_x, total_error) == 0), (
            'Total error anticommutes with logical X'
        )
        assert np.all(bs_prod(code.logicals_z, total_error) == 0), (
            'Total error anticommutes with logical Z'
        )

    @pytest.mark.parametrize('locations', [
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
        'up_left_horizontal_bottom',
        'up_left_horizontal_top',
        'down_right_horizontal',
    ])
    def test_errors_spanning_boundaries(self, code, decoder, locations):
        error = code.to_bsf({
            location: pauli
            for pauli, location in locations
        })
        assert bsf_wt(error) == len(locations)

        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        correction = decoder.decode(syndrome)
        total_error = (error + correction) % 2
        assert not np.all(
            code.measure_syndrome(total_error) == 0
        ), 'Total error in codespace when it should not be'


class TestSweepCorners(DecoderTest):
    """Test 1-qubit errors on corners fully correctable."""

    @pytest.fixture
    def code(self):
        return RotatedPlanar3DCode(5, 5, 3)

    @pytest.fixture
    def decoder(self, code, error_model):
        error_rate = 0.5
        return RotatedSweepMatchDecoder(code, error_model, error_rate)

    @pytest.mark.parametrize('location', [
        (1, 3, 5),
        (3, 5, 5),
        (5, 7, 5),
        (7, 9, 5),
        (9, 9, 5)
    ])
    def test_sweep_errors_on_extreme_layer(self, code, decoder, location):
        error = code.to_bsf({
            location: 'Z'
        })
        assert bsf_wt(error) == 1

        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        correction = decoder.decode(syndrome)
        total_error = (error + correction) % 2
        assert np.all(code.measure_syndrome(total_error) == 0), (
            'Total error not in codespace'
        )

        assert np.all(bs_prod(code.logicals_x, total_error) == 0), (
            'Total error anticommutes with logical X'
        )
        assert np.all(bs_prod(code.logicals_z, total_error) == 0), (
            'Total error anticommutes with logical Z'
        )

    @pytest.mark.parametrize('pauli', ['X', 'Y', 'Z'])
    def test_all_1_qubit_errors_correctable(self, code, decoder, pauli):
        uncorrectable_locations = []
        for location in code.qubit_coordinates:
            error = code.to_bsf({
                location: pauli
            })
            assert bsf_wt(error) == 1

            syndrome = code.measure_syndrome(error)
            assert np.any(syndrome != 0)

            correction = decoder.decode(syndrome)
            total_error = (error + correction) % 2
            assert np.all(code.measure_syndrome(total_error) == 0)

            correctable = True
            if np.any(bs_prod(code.logicals_x, total_error) != 0):
                correctable = False
            if np.all(bs_prod(code.logicals_z, total_error) != 0):
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
            error = code.to_bsf({
                location: pauli
                for location in locations
            })
            assert bsf_wt(error) == len(locations)

            syndrome = code.measure_syndrome(error)
            assert np.any(syndrome != 0)

            correction = decoder.decode(syndrome)
            total_error = (error + correction) % 2
            assert np.all(code.measure_syndrome(total_error) == 0)

            correctable = True
            if np.any(bs_prod(code.logicals_x, total_error) != 0):
                correctable = False
            if np.all(bs_prod(code.logicals_z, total_error) != 0):
                correctable = False
            if not correctable:
                uncorrectable_error_locations.append(locations)

        assert len(uncorrectable_error_locations) == 0, (
            f'Found {len(uncorrectable_error_locations)} uncorrectable '
            f'weight-2 Z errors'
        )


class TestRotatedSweepDecoder3D(DecoderTest):

    @pytest.fixture
    def code(self):
        return RotatedPlanar3DCode(3, 3, 3)

    @pytest.fixture
    def decoder(self, code, error_model):
        error_rate = 0.5
        return RotatedSweepDecoder3D(code, error_model, error_rate)

    @pytest.fixture
    def allowed_paulis(self):
        return ['Z']

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
        assert code.stabilizer_type(vertex) == 'vertex'
        x_face, y_face, z_face = decoder.get_sweep_faces(
            vertex, sweep_direction
        )
        x_edge, y_edge, z_edge = decoder.get_sweep_edges(
            vertex, sweep_direction
        )
        assert [x_face, y_face, z_face] == sweep_faces
        assert [x_edge, y_edge, z_edge] == sweep_edges

    def test_sweep_touches_all_faces_and_qubits(self, code, decoder):
        touched_edges = []
        touched_faces = []
        sweep_directions = [
            (1, 0, 1), (1, 0, -1),
            (0, 1, 1), (0, 1, -1),
            (-1, 0, 1), (-1, 0, -1),
            (0, -1, 1), (0, -1, -1),
        ]
        vertices = [
            location
            for location in code.stabilizer_coordinates
            if code.stabilizer_type(location) == 'vertex'
        ]
        face_index = {
            location: index
            for index, location in enumerate(code.stabilizer_coordinates)
            if code.stabilizer_type(location) == 'face'
        }
        for sweep_direction in sweep_directions:
            for vertex in vertices:
                x_face, y_face, z_face = decoder.get_sweep_faces(
                    vertex, sweep_direction
                )
                x_edge, y_edge, z_edge = decoder.get_sweep_edges(
                    vertex, sweep_direction
                )

                faces_valid = tuple(
                    face in face_index
                    for face in [x_face, y_face, z_face]
                )
                edges_valid = tuple(
                    edge in code.qubit_coordinates
                    for edge in [x_edge, y_edge, z_edge]
                )
                if all(faces_valid) and all(edges_valid):
                    x_face_bsf = code.stabilizer_matrix[face_index[x_face]]
                    y_face_bsf = code.stabilizer_matrix[face_index[y_face]]
                    z_face_bsf = code.stabilizer_matrix[face_index[z_face]]

                    error = dict()
                    error[x_edge] = 'Z'
                    x_edge_bsf = code.to_bsf(error)
                    assert np.all(bs_prod(x_edge_bsf, x_face_bsf) == 0)
                    assert np.any(bs_prod(x_edge_bsf, y_face_bsf) == 1)
                    assert np.any(bs_prod(x_edge_bsf, z_face_bsf) == 1)

                    error = dict()
                    error[y_edge] = 'Z'
                    y_edge_bsf = code.to_bsf(error)
                    assert np.all(bs_prod(y_edge_bsf, y_face_bsf) == 0)
                    assert np.any(bs_prod(y_edge_bsf, x_face_bsf) == 1)
                    assert np.any(bs_prod(y_edge_bsf, z_face_bsf) == 1)

                    error = dict()
                    error[z_edge] = 'Z'
                    z_edge_bsf = code.to_bsf(error)
                    assert np.any(bs_prod(z_edge_bsf, z_face_bsf) == 0)
                    assert np.any(bs_prod(z_edge_bsf, x_face_bsf) == 1)
                    assert np.any(bs_prod(z_edge_bsf, y_face_bsf) == 1)

                    touched_edges.append(x_edge)
                    touched_edges.append(y_edge)
                    touched_edges.append(z_edge)

                    touched_faces.append(x_face)
                    touched_faces.append(y_face)
                    touched_faces.append(z_face)

        assert set(code.qubit_coordinates) == set(touched_edges)
        assert set(face_index.keys()) == set(touched_faces)

    def test_flip_edge(self, code, decoder):
        for edge in code.qubit_coordinates:
            signs = {
                location: 0
                for location in code.stabilizer_coordinates
                if code.stabilizer_type(location) == 'face'
            }
            signs = decoder.get_initial_state(
                np.zeros(code.stabilizer_matrix.shape[0], dtype=np.uint)
            )
            decoder.flip_edge(edge, signs)
            sign_flip_syndrome = signs

            error = code.to_bsf({
                edge: 'Z'
            })
            pauli_syndrome = code.measure_syndrome(error)

            assert np.all(pauli_syndrome == sign_flip_syndrome)
