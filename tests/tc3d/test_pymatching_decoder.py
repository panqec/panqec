import itertools
import pytest
import numpy as np
from bn3d.bpauli import bcommute, bsf_wt
from bn3d.models import Toric3DCode, Toric3DPauli
from bn3d.decoders import Toric3DPymatchingDecoder


class TestToric3DPymatchingDecoder:

    @pytest.fixture
    def code(self):
        return Toric3DCode(3, 4, 5)

    @pytest.fixture
    def decoder(self):
        return Toric3DPymatchingDecoder()

    def test_decoder_has_required_attributes(self, decoder):
        assert decoder.label is not None
        assert decoder.decode is not None

    def test_decode_trivial_syndrome(self, decoder, code):
        syndrome = np.zeros(shape=code.stabilizers.shape[0], dtype=np.uint)
        correction = decoder.decode(code, syndrome)
        assert correction.shape == 2*code.n
        assert np.all(bcommute(code.stabilizers, correction) == 0)
        assert issubclass(correction.dtype.type, np.integer)

    def test_decode_X_error(self, decoder, code):
        error = Toric3DPauli(code)
        error.site('X', (2, 1, 2))
        assert bsf_wt(error.to_bsf()) == 1

        # Measure the syndrome and ensure non-triviality.
        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        correction = decoder.decode(code, syndrome)
        total_error = (error.to_bsf() + correction) % 2

        assert np.all(bcommute(code.stabilizers, total_error) == 0)

    def test_decode_many_X_errors(self, decoder, code):
        error = Toric3DPauli(code)
        error.site('X', (1, 0, 0))
        error.site('X', (0, 1, 0))
        error.site('X', (0, 0, 3))
        assert bsf_wt(error.to_bsf()) == 3

        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        correction = decoder.decode(code, syndrome)
        total_error = (error.to_bsf() + correction) % 2
        assert np.all(bcommute(code.stabilizers, total_error) == 0)

    def test_unable_to_decode_Z_error(self, decoder, code):
        error = Toric3DPauli(code)
        error.site('Z', (1, 0, 2))

        assert bsf_wt(error.to_bsf()) == 1

        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        correction = decoder.decode(code, syndrome)
        assert np.all(correction == 0)

        total_error = (error.to_bsf() + correction) % 2
        assert np.all(error.to_bsf() == total_error)

        assert np.any(bcommute(code.stabilizers, total_error) != 0)

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
            error = Toric3DPauli(code)
            error.site('X', site)
            syndrome = code.measure_syndrome(error)
            correction = decoder.decode(code, syndrome)
            total_error = (error.to_bsf() + correction) % 2
            assert np.all(bcommute(code.stabilizers, total_error) == 0)
