import itertools
import pytest
import numpy as np
from panqec.bpauli import bcommute, bsf_wt
from panqec.codes import Toric3DCode
from panqec.decoders import Toric3DPymatchingDecoder
from panqec.error_models import PauliErrorModel


class TestToric3DPymatchingDecoder:

    @pytest.fixture
    def code(self):
        return Toric3DCode(3, 4, 5)

    @pytest.fixture
    def decoder(self):
        error_model = PauliErrorModel(1/3, 1/3, 1/3)
        probability = 0.1
        return Toric3DPymatchingDecoder(error_model, probability)

    def test_decoder_has_required_attributes(self, decoder):
        assert decoder.label is not None
        assert decoder.decode is not None

    def test_decode_trivial_syndrome(self, decoder, code):
        syndrome = np.zeros(
            shape=code.stabilizer_matrix.shape[0], dtype=np.uint
        )
        correction = decoder.decode(code, syndrome)
        assert correction.shape[0] == 2*code.n
        assert np.all(bcommute(code.stabilizer_matrix, correction) == 0)
        assert issubclass(correction.dtype.type, np.integer)

    def test_decode_X_error(self, decoder, code):
        error = code.to_bsf({
            (2, 1, 2): 'X',
        })
        assert bsf_wt(error) == 1

        # Measure the syndrome and ensure non-triviality.
        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        correction = decoder.decode(code, syndrome)
        total_error = (error + correction) % 2

        assert np.all(bcommute(code.stabilizer_matrix, total_error) == 0)

    def test_decode_many_X_errors(self, decoder, code):
        error = code.to_bsf({
            (1, 0, 0): 'X',
            (0, 1, 0): 'X',
            (0, 0, 3): 'X',
        })
        assert bsf_wt(error) == 3

        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        correction = decoder.decode(code, syndrome)
        total_error = (error + correction) % 2
        assert np.all(bcommute(code.stabilizer_matrix, total_error) == 0)

    def test_unable_to_decode_Z_error(self, decoder, code):
        error = code.to_bsf({
            (1, 0, 2): 'Z'
        })

        assert bsf_wt(error) == 1

        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        correction = decoder.decode(code, syndrome)
        assert np.all(correction == 0)

        total_error = (error + correction) % 2
        assert np.all(error == total_error)

        assert np.any(bcommute(code.stabilizer_matrix, total_error) != 0)

    def test_decode_many_codes_and_errors_with_same_decoder(self, decoder):

        codes = [
            Toric3DCode(3, 4, 5),
            Toric3DCode(3, 3, 3),
            Toric3DCode(5, 4, 3),
        ]

        sites = [
            (0, 0, 1),
            (1, 0, 0),
            (0, 1, 0)
        ]

        for code, site in itertools.product(codes, sites):
            error = code.to_bsf({
                site: 'X',
            })
            syndrome = code.measure_syndrome(error)
            correction = decoder.decode(code, syndrome)
            total_error = (error + correction) % 2
            assert np.all(bcommute(code.stabilizer_matrix, total_error) == 0)
