import itertools
import pytest
import numpy as np
from qecsim.paulitools import bsf_wt
from bn3d.bpauli import bcommute
from bn3d.tc3d import ToricCode3D, SweepMatchDecoder, Toric3DPauli


class TestSweepMatchDecoder:

    @pytest.fixture
    def code(self):
        return ToricCode3D(3, 4, 5)

    @pytest.fixture
    def decoder(self):
        return SweepMatchDecoder()

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
            ('X', (0, 2, 2, 2)),
            ('Y', (1, 2, 0, 3)),
            ('Z', (2, 1, 2, 3)),
        ]
    )
    def test_decode_single_error(self, decoder, code, pauli, location):
        error = Toric3DPauli(code)
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
        error = Toric3DPauli(code)
        for pauli, location in paulis_locations:
            error.site(pauli, location)
        assert bsf_wt(error.to_bsf()) == len(paulis_locations)

        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        correction = decoder.decode(code, syndrome)
        total_error = (error.to_bsf() + correction) % 2
        assert np.all(bcommute(code.stabilizers, total_error) == 0)

    @pytest.mark.skip
    def test_decode_many_codes_and_errors_with_same_decoder(self, decoder):

        codes = [
            ToricCode3D(3, 4, 5),
            ToricCode3D(3, 3, 3),
            ToricCode3D(5, 4, 3),
        ]

        sites = [
            (1, 2, 2, 2),
            (0, 1, 0, 2),
            (2, 1, 1, 1)
        ]

        for code, site in itertools.product(codes, sites):
            error = Toric3DPauli(code)
            error.site('Z', site)
            syndrome = code.measure_syndrome(error)
            correction = decoder.decode(code, syndrome)
            total_error = (error.to_bsf() + correction) % 2
            assert np.all(bcommute(code.stabilizers, total_error) == 0)
