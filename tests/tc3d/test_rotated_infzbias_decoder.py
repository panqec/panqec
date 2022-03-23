import pytest
from itertools import combinations
import numpy as np
from qecsim.paulitools import bsf_wt
from bn3d.bpauli import bcommute
from bn3d.decoders import split_posts_at_active_fences
from bn3d.models import RotatedPlanar3DCode, RotatedPlanar3DPauli
from bn3d.decoders import RotatedInfiniteZBiasDecoder


@pytest.mark.parametrize('active_fences, segments, n_fences', [
    (
        [],
        [[0, 1, 2, 3, 4, 5, 6]],
        6
    ),
    (
        [0],
        [[0], [1, 2, 3, 4, 5, 6]],
        6
    ),
    (
        [5],
        [[0, 1, 2, 3, 4, 5], [6]],
        6
    ),
    (
        [2, 4],
        [[0, 1, 2], [3, 4], [5, 6]],
        6
    ),
])
def test_split_posts_at_active_fences_trivial(
    active_fences, segments, n_fences
):
    assert segments == split_posts_at_active_fences(active_fences, n_fences)


@pytest.mark.skip(reason='sparse')
class TestRotatedInfiniteZBiasDecoder:
    """Test 1-qubit errors on corners fully correctable."""

    @pytest.fixture
    def code(self):
        return RotatedPlanar3DCode(2, 2, 2)

    @pytest.fixture
    def decoder(self):
        return RotatedInfiniteZBiasDecoder()

    @pytest.mark.parametrize('location', [
        (1, 3, 5),
        (3, 5, 5),
        (5, 7, 5),
        (7, 9, 5),
        (9, 9, 5)
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
        assert np.all(bcommute(code.stabilizers, total_error) == 0), (
            'Total error not in codespace'
        )

        assert np.all(bcommute(code.logicals_x, total_error) == 0), (
            'Total error anticommutes with logical X'
        )
        assert np.all(bcommute(code.logicals_z, total_error) == 0), (
            'Total error anticommutes with logical Z'
        )

    @pytest.mark.parametrize('pauli', ['X', 'Z'])
    def test_all_1_qubit_errors_correctable(self, code, decoder, pauli):
        uncorrectable_locations = []

        # Filter down allowable errors for infinite bias XZZX deformed noise.
        if pauli == 'X':
            qubit_locations = [
                (x, y, z) for x, y, z in code.qubit_index
                if z % 2 == 0
            ]
        else:
            qubit_locations = [
                (x, y, z) for x, y, z in code.qubit_index
                if z % 2 == 1
            ]

        for location in qubit_locations:
            error = RotatedPlanar3DPauli(code)
            error.site(pauli, location)
            assert bsf_wt(error.to_bsf()) == 1

            syndrome = code.measure_syndrome(error)
            assert np.any(syndrome != 0)

            correction = decoder.decode(code, syndrome)
            total_error = (error.to_bsf() + correction) % 2

            assert np.all(bcommute(code.stabilizers, total_error) == 0)

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

    def test_all_2_qubit_errors_correctable(self, code, decoder):
        pauli = 'Z'
        weight = 2

        # Filter out z edges, which are not in the loop sector.
        edge_qubits = [
            (x, y, z) for x, y, z in code.qubit_index
            if z % 2 != 0
        ]
        error_locations = combinations(edge_qubits, weight)
        error_locations = [((1, 1, 3), (3, 1, 1))]

        uncorrectable_error_locations = []
        undecodable_error_locations = []
        for locations in error_locations:
            error = RotatedPlanar3DPauli(code)
            for location in locations:
                error.site(pauli, location)
            assert bsf_wt(error.to_bsf()) == len(locations)

            syndrome = code.measure_syndrome(error)
            assert np.any(syndrome != 0)

            correction = decoder.decode(code, syndrome)
            total_error = (error.to_bsf() + correction) % 2

            decodable = True
            if np.any(bcommute(code.stabilizers, total_error) != 0):
                decodable = False
            if not decodable:
                undecodable_error_locations.append(locations)

            correctable = True
            if np.any(bcommute(code.logicals_x, total_error) != 0):
                correctable = False
            if np.all(bcommute(code.logicals_z, total_error) != 0):
                correctable = False
            if not correctable:
                uncorrectable_error_locations.append(locations)

        assert len(undecodable_error_locations) == 0, (
            f'Found {len(undecodable_error_locations)} undecodable '
            f'weight-2 Z errors'
        )

        assert len(uncorrectable_error_locations) == 0, (
            f'Found {len(uncorrectable_error_locations)} uncorrectable '
            f'weight-2 Z errors'
        )
