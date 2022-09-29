import pytest
import numpy as np
from panqec.codes import Toric2DCode, Toric3DCode
from panqec.bpauli import bsf_wt
from panqec.decoders import MatchingDecoder
from panqec.error_models import PauliErrorModel
from tests.decoders.decoder_test import DecoderTest


class TestToric2DMatchingDecoder(DecoderTest):

    @pytest.fixture
    def code(self):
        return Toric2DCode(4, 5)

    @pytest.fixture
    def decoder(self, code, error_model):
        error_rate = 0.1
        return MatchingDecoder(code, error_model, error_rate)

    @pytest.mark.parametrize(
        'operator, location',
        [
            ('X', (0, 1)),
            ('Y', (0, 1)),
            ('Z', (0, 1)),
            ('X', (1, 0)),
            ('Y', (1, 0)),
            ('Z', (1, 0)),
        ]
    )
    def test_decode_single_error(self, code, operator, location):
        error_model = PauliErrorModel(1/3, 1/3, 1/3)
        error_rate = 0.5
        decoder = MatchingDecoder(code, error_model, error_rate)
        error = code.to_bsf({
            location: operator
        })
        assert bsf_wt(error) == 1, 'Error should be weight 1'
        syndromes = code.measure_syndrome(error)
        if operator in ['Z', 'X']:
            assert sum(syndromes) == 2, 'Should be 2 syndromes if X or Z error'
        elif operator == 'Y':
            assert sum(syndromes) == 4, 'Should be 4 syndromes if Y error'
        correction = decoder.decode(syndromes)
        assert bsf_wt(correction) == 1, 'Correction should be weight 1'
        total_error = (error + correction) % 2
        assert np.all(code.measure_syndrome(total_error) == 0), (
            'Total error should be in code space'
        )

    def test_exception_when_wrong_code(self, code):
        error_model = PauliErrorModel(1/3, 1/3, 1/3)
        error_rate = 0.5
        MatchingDecoder(code, error_model, error_rate)

        code_3d = Toric3DCode(3)
        with pytest.raises(Exception):
            MatchingDecoder(code_3d, error_model, error_rate)
