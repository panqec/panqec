import itertools
import pytest
import numpy as np
from qecsim.paulitools import bsf_wt
from bn3d.bpauli import bcommute
from bn3d.tc3d import (
    RotatedPlanarCode3D, RotatedSweepMatchDecoder, RotatedPlanar3DPauli
)


class TestRotatedSweepMatchDecoder:

    @pytest.fixture
    def code(self):
        return RotatedPlanarCode3D(3, 4, 5)

    @pytest.fixture
    def decoder(self):
        return RotatedSweepMatchDecoder()

    def test_decoder_has_required_attributes(self, decoder):
        assert decoder.label is not None
        assert decoder.decode is not None

    def test_decode_trivial_syndrome(self, decoder, code):
        syndrome = np.zeros(shape=len(code.stabilizers), dtype=np.uint)
        correction = decoder.decode(code, syndrome)
        assert correction.shape == 2*code.n_k_d[0]
        assert np.all(bcommute(code.stabilizers, correction) == 0)
        assert issubclass(correction.dtype.type, np.integer)

    @pytest.mark.parametrize(
        'pauli, location',
        [
            ('X', (3, 3, 1)),
            ('Y', (7, 9, 5)),
            ('Z', (6, 4, 10)),
        ]
    )
    def test_decode_single_error(self, decoder, code, pauli, location):
        error = RotatedPlanar3DPauli(code)
        error.site(pauli, location)
        assert bsf_wt(error.to_bsf()) == 1

        # Measure the syndrome and ensure non-triviality.
        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        correction = decoder.decode(code, syndrome)
        total_error = (error.to_bsf() + correction) % 2
        assert np.all(bcommute(code.stabilizers, total_error) == 0)

    @pytest.mark.parametrize(
        'paulis_locations',
        [
            [
                ('X', (3, 3, 1)),
                ('Y', (7, 9, 5)),
                ('Z', (6, 4, 10)),
            ],
            [
                ('Y', (9, 5, 1)),
                ('Y', (2, 12, 4)),
                ('Y', (6, 8, 4)),
            ],
        ]
    )
    def test_decode_many_errors(self, decoder, code, paulis_locations):
        error = RotatedPlanar3DPauli(code)
        for pauli, location in paulis_locations:
            error.site(pauli, location)
        assert bsf_wt(error.to_bsf()) == len(paulis_locations)

        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        correction = decoder.decode(code, syndrome)
        total_error = (error.to_bsf() + correction) % 2
        assert np.all(bcommute(code.stabilizers, total_error) == 0)

    def test_decode_many_codes_and_errors_with_same_decoder(self, decoder):

        codes_sites = [
            (RotatedPlanarCode3D(3, 4, 5), (7, 9, 3)),
            (RotatedPlanarCode3D(3, 3, 3), (3, 5, 7)),
            (RotatedPlanarCode3D(5, 4, 3), (1, 3, 5)),
        ]

        for code, site in codes_sites:
            error = RotatedPlanar3DPauli(code)
            error.site('Z', site)
            syndrome = code.measure_syndrome(error)
            correction = decoder.decode(code, syndrome)
            total_error = (error.to_bsf() + correction) % 2
            assert np.all(bcommute(code.stabilizers, total_error) == 0)
