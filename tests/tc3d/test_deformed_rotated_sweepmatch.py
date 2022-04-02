from typing import Tuple
import numpy as np
import pytest
from panqec.codes import RotatedPlanar3DCode
from panqec.decoders import DeformedRotatedSweepMatchDecoder
from panqec.error_models import DeformedXZZXErrorModel
from panqec.bpauli import bcommute, get_effective_error


class TestDeformedRotatedPlanarPymatchingDecoder:

    direction: Tuple[float, float, float] = 0.05, 0.05, 0.9
    size: Tuple[int, int, int] = 2, 2, 2
    probability: float = 0.1

    @pytest.fixture
    def code(self):
        return RotatedPlanar3DCode(*self.size)

    @pytest.fixture
    def error_model(self):
        return DeformedXZZXErrorModel(*self.direction)

    @pytest.fixture
    def decoder(self, error_model):
        return DeformedRotatedSweepMatchDecoder(error_model, self.probability)

    def test_decode(self, code, error_model, decoder):
        rng = np.random.default_rng(seed=0)
        error = error_model.generate(code, self.probability, rng=rng)
        assert np.any(error != 0)

        syndrome = bcommute(code.stabilizer_matrix, error)
        correction = decoder.decode(code, syndrome)
        total_error = (correction + error) % 2
        effective_error = get_effective_error(
            total_error, code.logicals_x, code.logicals_z
        )
        codespace = bool(np.all(bcommute(code.stabilizer_matrix, total_error) == 0))
        assert codespace, 'Not in code space'
        success = bool(np.all(effective_error == 0)) and codespace
        assert success, 'Decoding failed'

    def test_decode_along_line_preferred(self, code, error_model, decoder):

        # Two close-together parallel lines of X errors along deformation axis.
        pauli_error = dict()
        pauli_error[(4, 2, 2)] = 'X'
        pauli_error[(4, 2, 4)] = 'X'
        pauli_error[(6, 4, 2)] = 'X'
        pauli_error[(6, 4, 4)] = 'X'
        error = code.to_bsf(pauli_error)

        # The expected correction, given the high bias, should be matching
        # along the deformation axis, which is the original error itself.
        expected_correction = error

        # (as opposed to the naive correction of joining the two ends across to
        # form a total error that is a loop)
        pauli_naive_correction = dict()
        pauli_naive_correction[(5, 3, 1)] = 'X'
        pauli_naive_correction[(5, 3, 5)] = 'X'
        naive_correction = code.to_bsf(pauli_naive_correction)

        syndrome = bcommute(code.stabilizer_matrix, error)
        correction = decoder.decode(code, syndrome)

        assert np.any(correction != naive_correction), 'Correction is naive'
        assert np.all(correction == expected_correction), (
            'Correction not as expected'
        )
        total_error = (correction + error) % 2

        effective_error = get_effective_error(
            total_error, code.logicals_x, code.logicals_z
        )
        codespace = bool(np.all(bcommute(code.stabilizer_matrix, total_error) == 0))
        assert codespace, 'Not in code space'
        success = bool(np.all(effective_error == 0)) and codespace
        assert success, 'Decoding failed'
