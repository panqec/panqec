import pytest
import numpy as np
from qecsim.paulitools import bsf_wt
from bn3d.bpauli import bcommute
from bn3d.tc3d import (
    RotatedPlanarCode3D, RotatedSweepMatchDecoder, RotatedPlanar3DPauli
)


class TestRotatedSweepMatchDecoder:

    @pytest.fixture
    def code(self):
        return RotatedPlanarCode3D(4, 4, 4)

    @pytest.fixture
    def decoder(self):
        return RotatedSweepMatchDecoder()

    def test_decoder_has_required_attributes(self, decoder):
        assert decoder.label is not None
        assert decoder.decode is not None

    def test_decode_trivial_syndrome(self, decoder, code):
        syndrome = np.zeros(shape=len(code.stabilizers), dtype=np.uint)
        correction = decoder.decode(code, syndrome)
        assert correction.shape[0] == 2*code.n_k_d[0]
        assert np.all(bcommute(code.stabilizers, correction) == 0)
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
        error = RotatedPlanar3DPauli(code)
        assert location in code.qubit_index
        error.site(pauli, location)
        assert bsf_wt(error.to_bsf()) == 1

        # Measure the syndrome and ensure non-triviality.
        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        correction = decoder.decode(code, syndrome)
        total_error = (error.to_bsf() + correction) % 2
        assert np.all(bcommute(code.stabilizers, total_error) == 0)

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
        error = RotatedPlanar3DPauli(code)
        for pauli, location in paulis_locations:
            assert location in code.qubit_index.keys()
            error.site(pauli, location)
        assert bsf_wt(error.to_bsf()) == len(paulis_locations)

        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        correction = decoder.decode(code, syndrome)
        total_error = (error.to_bsf() + correction) % 2
        assert np.all(bcommute(code.stabilizers, total_error) == 0)

    def test_undecodable_error(self, decoder, code):
        locations = [
            (x, y, z) for x, y, z in code.qubit_index
        ]
        assert len(locations) > 0
        error = RotatedPlanar3DPauli(code)
        for location in locations:
            assert location in code.qubit_index.keys()
            error.site('Z', location)
        assert bsf_wt(error.to_bsf()) == len(locations)

        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        correction = decoder.decode(code, syndrome)
        total_error = (error.to_bsf() + correction) % 2
        assert np.any(total_error)

    def test_decode_many_codes_and_errors_with_same_decoder(self, decoder):

        codes_sites = [
            (RotatedPlanarCode3D(3, 3, 3), (7, 9, 3)),
            (RotatedPlanarCode3D(4, 4, 4), (3, 5, 7)),
            (RotatedPlanarCode3D(5, 5, 5), (1, 3, 5)),
        ]

        for code, site in codes_sites:
            error = RotatedPlanar3DPauli(code)
            assert site in code.qubit_index.keys()
            error.site('Z', site)
            syndrome = code.measure_syndrome(error)
            correction = decoder.decode(code, syndrome)
            total_error = (error.to_bsf() + correction) % 2
            assert np.all(bcommute(code.stabilizers, total_error) == 0)


class TestSweepMatch1x1x1:
    """Test cases found to be failing on the GUI."""

    @pytest.fixture
    def code(self):
        return RotatedPlanarCode3D(1, 1, 1)

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
        error = RotatedPlanar3DPauli(code)
        for pauli, location in locations:
            error.site(pauli, location)
        assert bsf_wt(error.to_bsf()) == len(locations)

        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        correction = decoder.decode(code, syndrome)
        total_error = (error.to_bsf() + correction) % 2
        assert np.all(bcommute(code.stabilizers, total_error) == 0)


class TestSweepMatch2x2x2:
    """Test cases found to be failing on the GUI."""

    @pytest.fixture
    def code(self):
        return RotatedPlanarCode3D(2, 2, 2)

    @pytest.fixture
    def decoder(self):
        return RotatedSweepMatchDecoder()

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
        [
            ('Z', (1, 1, 5)), ('Z', (3, 3, 5)), ('Z', (5, 5, 5)),
            ('Z', (7, 5, 5)), ('Z', (9, 3, 5)),
        ],
    ], ids=[
        'z_vertical',
        'up_left_horizontal_bottom',
        'up_left_horizontal_top',
        'down_right_horizontal',
        'arthurs_example',
    ])
    def test_errors_spanning_boundaries(self, code, decoder, locations):
        error = RotatedPlanar3DPauli(code)
        for pauli, location in locations:
            assert location in code.qubit_index
            error.site(pauli, location)
        assert bsf_wt(error.to_bsf()) == len(locations)

        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        correction = decoder.decode(code, syndrome)
        total_error = (error.to_bsf() + correction) % 2
        assert np.all(bcommute(code.stabilizers, total_error) == 0)


class TestSweepCorners:
    """Test 1-qubit errors on corners fully correctable."""

    @pytest.fixture
    def code(self):
        return RotatedPlanarCode3D(2, 2, 2)

    @pytest.fixture
    def decoder(self):
        return RotatedSweepMatchDecoder()

    @pytest.mark.parametrize('location', [
        # (1, 3, 5),
        # (3, 5, 5),
        (5, 7, 5),
        # (7, 9, 5),
        # (9, 9, 5)
    ])
    def test_sweep_errors_on_extreme_layer(self, code, decoder, location):
        error = RotatedPlanar3DPauli(code)
        assert location in code.qubit_index
        error.site('Z', location)
        assert bsf_wt(error.to_bsf()) == 1

        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        correction = decoder.decode(code, syndrome)
        total_error = (error.to_bsf() + correction) % 2
        assert np.all(bcommute(code.stabilizers, total_error) == 0)

        assert np.all(bcommute(code.logical_xs, total_error) == 0)
        assert np.all(bcommute(code.logical_zs, total_error) == 0)

    @pytest.mark.parametrize('pauli', ['X', 'Y', 'Z'])
    def test_all_1_qubit_errors_correctable(self, code, decoder, pauli):
        uncorrectable_locations = []
        for location in code.qubit_index:
            error = RotatedPlanar3DPauli(code)
            error.site(pauli, location)
            assert bsf_wt(error.to_bsf()) == 1

            syndrome = code.measure_syndrome(error)
            assert np.any(syndrome != 0)

            correction = decoder.decode(code, syndrome)
            total_error = (error.to_bsf() + correction) % 2
            assert np.all(bcommute(code.stabilizers, total_error) == 0)

            correctable = True
            if np.any(bcommute(code.logical_xs, total_error) != 0):
                correctable = False
            if np.all(bcommute(code.logical_zs, total_error) != 0):
                correctable = False
            if not correctable:
                uncorrectable_locations.append(location)

        assert len(uncorrectable_locations) == 0
