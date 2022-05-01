import itertools
import pytest
import numpy as np
from panqec.bpauli import bcommute, bsf_wt
from panqec.codes import Toric3DCode
from panqec.decoders import SweepMatchDecoder
from panqec.error_models import PauliErrorModel


class TestSweepMatchDecoder:

    @pytest.fixture
    def code(self):
        return Toric3DCode(3, 4, 5)

    @pytest.fixture
    def decoder(self):
        error_model = PauliErrorModel(1/3, 1/3, 1/3)
        probability = 0.5
        return SweepMatchDecoder(error_model, probability)

    def test_decoder_has_required_attributes(self, decoder):
        assert decoder.label is not None
        assert decoder.decode is not None

    def test_decode_trivial_syndrome(self, decoder, code):
        syndrome = np.zeros(
            shape=code.stabilizer_matrix.shape[0], dtype=np.uint
        )
        correction = decoder.decode(code, syndrome)
        assert correction.shape == (2*code.n,)
        assert np.all(bcommute(code.stabilizer_matrix, correction) == 0)
        assert issubclass(correction.dtype.type, np.integer)

    @pytest.mark.parametrize(
        'pauli, location',
        [
            ('X', (5, 4, 4)),
            ('Y', (4, 1, 6)),
            ('Z', (2, 4, 7)),
        ]
    )
    def test_decode_single_error(self, decoder, code, pauli, location):
        error = code.to_bsf({
            location: pauli
        })
        assert bsf_wt(error) == 1

        # Measure the syndrome and ensure non-triviality.
        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        correction = decoder.decode(code, syndrome)
        total_error = (error + correction) % 2
        assert np.all(bcommute(code.stabilizer_matrix, total_error) == 0)

    @pytest.mark.parametrize(
        'paulis_locations',
        [
            [
                ('X', (5, 4, 4)),
                ('Y', (4, 1, 6)),
                ('Z', (2, 4, 7)),
            ],
            [
                ('Y', (5, 2, 2)),
                ('Y', (4, 1, 6)),
                ('Y', (2, 4, 7)),
            ],
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

        correction = decoder.decode(code, syndrome)
        total_error = (error + correction) % 2
        assert np.all(bcommute(code.stabilizer_matrix, total_error) == 0)

    def test_decode_many_codes_and_errors_with_same_decoder(self, decoder):

        codes = [
            Toric3DCode(3, 4, 5),
            Toric3DCode(3, 3, 3),
            Toric3DCode(5, 4, 3),
        ]

        sites = [
            (4, 5, 4),
            (3, 0, 4),
            (2, 2, 1)
        ]

        for code, site in itertools.product(codes, sites):
            error = code.to_bsf({
                site: 'Z'
            })
            syndrome = code.measure_syndrome(error)
            correction = decoder.decode(code, syndrome)
            total_error = (error + correction) % 2
            assert np.all(bcommute(code.stabilizer_matrix, total_error) == 0)
