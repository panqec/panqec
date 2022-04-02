import itertools
import pytest
import numpy as np
from qecsim.paulitools import bsf_wt
from panqec.bpauli import bcommute
from panqec.codes import Toric3DCode
from panqec.decoders import SweepMatchDecoder


@pytest.mark.skip(reason='sparse')
class TestSweepMatchDecoder:

    @pytest.fixture
    def code(self):
        return Toric3DCode(3, 4, 5)

    @pytest.fixture
    def decoder(self):
        return SweepMatchDecoder()

    def test_decoder_has_required_attributes(self, decoder):
        assert decoder.label is not None
        assert decoder.decode is not None

    def test_decode_trivial_syndrome(self, decoder, code):
        syndrome = np.zeros(shape=len(code.stabilizer_matrix), dtype=np.uint)
        correction = decoder.decode(code, syndrome)
        assert correction.shape == 2*code.n
        assert np.all(bcommute(code.stabilizer_matrix, correction) == 0)
        assert issubclass(correction.dtype.type, np.integer)

    @pytest.mark.parametrize(
        'pauli, location',
        [
            ('X', (0, 2, 2, 2)),
            ('Y', (1, 2, 0, 3)),
            ('Z', (2, 1, 2, 3)),
        ]
    )
    def test_decode_single_error(self, decoder, code, pauli, location):
        error = dict()
        error[location] = pauli
        assert bsf_wt(code.to_bsf(error)) == 1

        # Measure the syndrome and ensure non-triviality.
        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        correction = decoder.decode(code, syndrome)
        total_error = (code.to_bsf(error) + correction) % 2
        assert np.all(bcommute(code.stabilizer_matrix, total_error) == 0)

    @pytest.mark.parametrize(
        'paulis_locations',
        [
            [
                ('X', (0, 2, 2, 2)),
                ('Y', (1, 2, 0, 3)),
                ('Z', (2, 1, 2, 3)),
            ],
            [
                ('Y', (0, 2, 1, 1)),
                ('Y', (1, 2, 0, 3)),
                ('Y', (2, 1, 2, 3)),
            ],
        ]
    )
    def test_decode_many_errors(self, decoder, code, paulis_locations):
        error = dict()
        for pauli, location in paulis_locations:
            error[location] = pauli
        assert bsf_wt(code.to_bsf(error)) == len(paulis_locations)

        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        correction = decoder.decode(code, syndrome)
        total_error = (code.to_bsf(error) + correction) % 2
        assert np.all(bcommute(code.stabilizer_matrix, total_error) == 0)

    def test_decode_many_codes_and_errors_with_same_decoder(self, decoder):

        codes = [
            Toric3DCode(3, 4, 5),
            Toric3DCode(3, 3, 3),
            Toric3DCode(5, 4, 3),
        ]

        sites = [
            (1, 2, 2, 2),
            (0, 1, 0, 2),
            (2, 1, 1, 1)
        ]

        for code, site in itertools.product(codes, sites):
            error = dict()
            error[site] = 'Z'
            syndrome = code.measure_syndrome(error)
            correction = decoder.decode(code, syndrome)
            total_error = (code.to_bsf(error) + correction) % 2
            assert np.all(bcommute(code.stabilizer_matrix, total_error) == 0)
