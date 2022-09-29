import itertools
import pytest
import numpy as np
from panqec.bpauli import bsf_wt
from panqec.codes import Toric3DCode
from panqec.decoders import MatchingDecoder
from panqec.error_models import PauliErrorModel
from tests.decoders.decoder_test import DecoderTest


class TestToric3DMatchingDecoder(DecoderTest):

    @pytest.fixture
    def code(self):
        return Toric3DCode(3, 4, 5)

    @pytest.fixture
    def decoder(self, code, error_model):
        error_rate = 0.1
        return MatchingDecoder(code, error_model, error_rate, 'X')

    @pytest.fixture
    def allowed_paulis(self):
        return ['X']

    def test_decode_X_error(self, decoder, code):
        error = code.to_bsf({
            (2, 1, 2): 'X',
        })
        assert bsf_wt(error) == 1

        # Measure the syndrome and ensure non-triviality.
        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        correction = decoder.decode(syndrome)
        total_error = (error + correction) % 2

        assert np.all(code.measure_syndrome(total_error) == 0)

    def test_decode_many_X_errors(self, decoder, code):
        error = code.to_bsf({
            (1, 0, 0): 'X',
            (0, 1, 0): 'X',
            (0, 0, 3): 'X',
        })
        assert bsf_wt(error) == 3

        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        correction = decoder.decode(syndrome)
        total_error = (error + correction) % 2
        assert np.all(code.measure_syndrome(total_error) == 0)

    def test_unable_to_decode_Z_error(self, decoder, code):
        error = code.to_bsf({
            (1, 0, 2): 'Z'
        })

        assert bsf_wt(error) == 1

        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        correction = decoder.decode(syndrome)
        assert np.all(correction == 0)

        total_error = (error + correction) % 2
        assert np.all(error == total_error)

        assert np.any(code.measure_syndrome(total_error) != 0)

    def test_decode_many_codes_and_errors_with_same_decoder(self):

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

        error_model = PauliErrorModel(1/3, 1/3, 1/3)
        error_rate = 0.1

        for code, site in itertools.product(codes, sites):
            decoder = MatchingDecoder(code, error_model, error_rate, 'X')
            error = code.to_bsf({
                site: 'X',
            })
            syndrome = code.measure_syndrome(error)
            correction = decoder.decode(syndrome)
            total_error = (error + correction) % 2
            assert np.all(code.measure_syndrome(total_error) == 0)
