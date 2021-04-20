import numpy as np
import pytest
from qecsim.paulitools import bsf_wt
from bn3d.bpauli import bcommute
from bn3d.tc3d import Toric3DPymatchingDecoder, ToricCode3D, Toric3DPauli


class TestToric3DPymatchingDecoder:

    @pytest.fixture
    def code(self):
        return ToricCode3D(3, 4, 5)

    @pytest.fixture
    def decoder(self):
        return Toric3DPymatchingDecoder()

    def test_decoder_has_required_attributes(self, decoder):
        assert decoder.label is not None
        assert decoder.decode is not None

    def test_decode_trivial_syndrome(self, decoder, code):
        syndrome = np.zeros(shape=len(code.stabilizers), dtype=np.uint)
        correction = decoder.decode(code, syndrome)
        assert correction.shape == 2*code.n_k_d[0]
        assert np.all(bcommute(code.stabilizers, correction) == 0)

    def test_decode_X_error(self, decoder, code):
        error = Toric3DPauli(code)
        error.site('X', (0, 2, 2, 2))
        assert bsf_wt(error.to_bsf()) == 1

        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        correction = decoder.decode(code, syndrome)
        total_error = error.to_bsf() + correction
        assert np.all(bcommute(code.stabilizers, total_error) == 0)
