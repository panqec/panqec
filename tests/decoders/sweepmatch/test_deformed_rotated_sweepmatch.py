from typing import Tuple
import numpy as np
import pytest
from panqec.codes import RotatedPlanar3DCode
from panqec.decoders import RotatedSweepMatchDecoder
from panqec.error_models import PauliErrorModel
from tests.decoders.decoder_test import DecoderTest


class TestDeformedRotatedPlanarMatchingDecoder(DecoderTest):

    direction: Tuple[float, float, float] = 0.05, 0.05, 0.9
    size: Tuple[int, int, int] = 4, 4, 3
    error_rate: float = 0.1

    @pytest.fixture
    def code(self):
        return RotatedPlanar3DCode(*self.size)

    @pytest.fixture
    def error_model(self):
        return PauliErrorModel(
            *self.direction,
            deformation_name='XZZX',
            deformation_kwargs={'deformation_axis': 'z'}
        )

    @pytest.fixture
    def decoder(self, code, error_model):
        return RotatedSweepMatchDecoder(code, error_model,
                                        self.error_rate)

    def test_decode(self, code, error_model, decoder):
        rng = np.random.default_rng(seed=0)
        error = error_model.generate(code, self.error_rate, rng=rng)
        assert np.any(error != 0)

        syndrome = code.measure_syndrome(error)
        correction = decoder.decode(syndrome)
        total_error = (correction + error) % 2
        effective_error = code.logical_errors(total_error)
        codespace = bool(
            np.all(code.measure_syndrome(total_error) == 0)
        )
        assert codespace, 'Not in code space'
        success = bool(np.all(effective_error == 0)) and codespace
        assert success, 'Decoding failed'

    def test_decode_along_line_preferred(self, code, error_model, decoder):

        # Two close-together parallel lines of X errors along deformation axis.
        error = code.to_bsf({
            (4, 2, 2): 'X',
            (4, 2, 4): 'X',
            (6, 4, 2): 'X',
            (6, 4, 4): 'X',
        })

        # The expected correction, given the high bias, should be matching
        # along the deformation axis, which is the original error itself.
        expected_correction = error

        # (as opposed to the naive correction of joining the two ends across to
        # form a total error that is a loop)
        pauli_naive_correction = dict()
        pauli_naive_correction[(5, 3, 1)] = 'X'
        pauli_naive_correction[(5, 3, 5)] = 'X'
        naive_correction = code.to_bsf(pauli_naive_correction)

        syndrome = code.measure_syndrome(error)
        correction = decoder.decode(syndrome)

        assert np.any(correction != naive_correction), 'Correction is naive'
        assert np.all(correction == expected_correction), (
            'Correction not as expected'
        )
        total_error = (correction + error) % 2

        effective_error = code.logical_errors(total_error)
        codespace = code.in_codespace(total_error)
        assert codespace, 'Not in code space'
        success = bool(np.all(effective_error == 0)) and codespace
        assert success, 'Decoding failed'
