import itertools
import pytest
import numpy as np
from qecsim.paulitools import bsf_wt
from bn3d.tc3d import ToricCode3D, SweepDecoder3D, Toric3DPauli
from bn3d.bpauli import bcommute
from bn3d.noise import PauliErrorModel


class TestSweepDecoder3D:

    @pytest.fixture
    def code(self):
        return ToricCode3D(3, 4, 5)

    @pytest.fixture
    def decoder(self):
        return SweepDecoder3D()

    def test_decoder_has_required_attributes(self, decoder):
        assert decoder.label is not None
        assert decoder.decode is not None

    def test_decode_trivial_syndrome(self, decoder, code):
        syndrome = np.zeros(shape=len(code.stabilizers), dtype=np.uint)
        correction = decoder.decode(code, syndrome)
        assert correction.shape == 2*code.n_k_d[0]
        assert np.all(bcommute(code.stabilizers, correction) == 0)
        assert issubclass(correction.dtype.type, np.integer)

    def test_decode_Z_error(self, decoder, code):
        error = Toric3DPauli(code)
        error.site('Z', (0, 2, 2, 2))
        assert bsf_wt(error.to_bsf()) == 1

        # Measure the syndrome and ensure non-triviality.
        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        correction = decoder.decode(code, syndrome)
        total_error = (error.to_bsf() + correction) % 2
        assert np.all(bcommute(code.stabilizers, total_error) == 0)

    def test_decode_many_Z_errors(self, decoder, code):
        error = Toric3DPauli(code)
        error.site('Z', (0, 2, 2, 2))
        error.site('Z', (1, 2, 2, 2))
        error.site('Z', (2, 2, 2, 2))
        assert bsf_wt(error.to_bsf()) == 3

        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        correction = decoder.decode(code, syndrome)
        total_error = (error.to_bsf() + correction) % 2
        assert np.all(bcommute(code.stabilizers, total_error) == 0)

    def test_unable_to_decode_X_error(self, decoder, code):
        error = Toric3DPauli(code)
        error.site('X', (0, 2, 2, 2))
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

    def test_decode_error_on_two_edges_sharing_same_vertex(self):
        code = ToricCode3D(3, 3, 3)
        decoder = SweepDecoder3D()
        error_pauli = Toric3DPauli(code)
        error_pauli.site('Z', (0, 1, 1, 1))
        error_pauli.site('Z', (1, 1, 1, 1))
        error = error_pauli.to_bsf()
        syndrome = bcommute(code.stabilizers, error)
        correction = decoder.decode(code, syndrome)
        total_error = (error + correction) % 2
        assert np.all(bcommute(code.stabilizers, total_error) == 0)

    def test_decode_with_general_Z_noise(self):
        code = ToricCode3D(3, 3, 3)
        decoder = SweepDecoder3D()
        np.random.seed(0)
        error_model = PauliErrorModel(0, 0, 1)

        in_codespace = []
        for i in range(100):
            error = error_model.generate(code, probability=0.01, rng=np.random)
            syndrome = bcommute(code.stabilizers, error)
            correction = decoder.decode(code, syndrome)
            total_error = (error + correction) % 2
            in_codespace.append(
                np.all(bcommute(code.stabilizers, total_error) == 0)
            )
        assert all(in_codespace)

    def test_sweep_move_two_edges(self):
        code = ToricCode3D(3, 3, 3)
        decoder = SweepDecoder3D()

        error = Toric3DPauli(code)
        error.site('Z', (0, 1, 1, 1))
        error.site('Z', (1, 1, 1, 1))

        syndrome = bcommute(code.stabilizers, error.to_bsf())

        correction = Toric3DPauli(code)

        # Syndrome from errors on x edge and y edge on vertex (0, 0, 0).
        signs = np.zeros((3, 3, 3, 3), dtype=np.uint)
        signs[1, 1, 1, 1] = 1
        signs[1, 1, 1, 0] = 1
        signs[0, 1, 1, 1] = 1
        signs[0, 1, 1, 0] = 1
        signs[2, 1, 0, 1] = 1
        signs[2, 0, 1, 1] = 1

        n_faces = code.n_k_d[0]
        assert np.all(syndrome[:n_faces].reshape(signs.shape) == signs)

        # Expected signs after one sweep.
        expected_signs_1 = np.zeros((3, 3, 3, 3), dtype=np.uint)
        expected_signs_1[2, 1, 0, 1] = 1
        expected_signs_1[2, 0, 1, 1] = 1
        expected_signs_1[0, 1, 0, 1] = 1
        expected_signs_1[0, 1, 0, 0] = 1
        expected_signs_1[1, 0, 1, 1] = 1
        expected_signs_1[1, 0, 1, 0] = 1
        signs_1 = decoder.sweep_move(signs, correction, default_direction=0)
        assert np.all(expected_signs_1 == signs_1)

        # Expected signs after two sweeps, should be all gone.
        signs_2 = decoder.sweep_move(signs_1, correction, default_direction=0)
        assert np.all(signs_2 == 0)

        expected_correction = Toric3DPauli(code)
        expected_correction.site('Z', (2, 1, 1, 1))
        expected_correction.site('Z', (0, 0, 1, 1))
        expected_correction.site('Z', (1, 1, 0, 1))
        expected_correction.site('Z', (2, 1, 1, 0))

        # Only need to compare the Z block because sweep only corrects Z block
        # anyway.
        correction_edges = set(
            map(tuple, np.array(np.where(correction._zs)).T)
        )
        expected_correction_edges = set(
            map(tuple, np.array(np.where(expected_correction._zs)).T)
        )

        assert correction_edges == expected_correction_edges
        assert np.all(correction._zs == expected_correction._zs)
