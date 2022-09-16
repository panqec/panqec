import pytest
from itertools import combinations
import numpy as np
from panqec.bpauli import bs_prod, bsf_wt
from panqec.decoders import split_posts_at_active_fences
from panqec.codes import RotatedPlanar3DCode
from panqec.decoders import RotatedInfiniteZBiasDecoder
from panqec.error_models import PauliErrorModel
from tests.decoders.decoder_test import DecoderTest


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


class TestRotatedInfiniteZBiasDecoder(DecoderTest):
    """Test 1-qubit errors on corners fully correctable."""

    @pytest.fixture
    def code(self):
        return RotatedPlanar3DCode(5, 5, 3)

    @pytest.fixture
    def error_model(self):
        return PauliErrorModel(0, 0, 1)

    @pytest.fixture
    def decoder(self, code, error_model):
        error_rate = 0.5
        return RotatedInfiniteZBiasDecoder(code, error_model, error_rate)

    @pytest.fixture
    def allowed_paulis(self):
        return ['Z']

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

    @pytest.mark.parametrize('pauli', ['X', 'Z'])
    def test_all_1_qubit_errors_correctable(self, code, decoder, pauli):
        uncorrectable_locations = []

        # Filter down allowable errors for infinite bias XZZX deformed noise.
        if pauli == 'X':
            qubit_locations = [
                (x, y, z) for x, y, z in code.qubit_coordinates
                if z % 2 == 0
            ]
        else:
            qubit_locations = [
                (x, y, z) for x, y, z in code.qubit_coordinates
                if z % 2 == 1
            ]

        for location in qubit_locations:
            error = code.to_bsf({
                location: pauli
            })
            assert bsf_wt(error) == 1

            syndrome = code.measure_syndrome(error)
            assert np.any(syndrome != 0)

            correction = decoder.decode(syndrome)
            total_error = (error + correction) % 2

            assert np.all(
                code.measure_syndrome(total_error) == 0
            ), 'Total error not in code space'

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

    def test_all_2_qubit_errors_correctable(self, code, decoder):
        pauli = 'Z'
        weight = 2

        # Filter out z edges, which are not in the loop sector.
        edge_qubits = [
            (x, y, z) for x, y, z in code.qubit_coordinates
            if z % 2 != 0
        ]
        error_locations = combinations(edge_qubits, weight)
        error_locations = [((1, 1, 3), (3, 1, 1))]

        uncorrectable_error_locations = []
        undecodable_error_locations = []
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

            decodable = True
            if np.any(code.measure_syndrome(total_error) != 0):
                decodable = False
            if not decodable:
                undecodable_error_locations.append(locations)

            correctable = True
            if np.any(bs_prod(code.logicals_x, total_error) != 0):
                correctable = False
            if np.all(bs_prod(code.logicals_z, total_error) != 0):
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
