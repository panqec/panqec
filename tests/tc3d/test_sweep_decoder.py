import itertools
import pytest
import numpy as np
from qecsim.paulitools import bsf_wt
from bn3d.models import Toric3DCode, Toric3DPauli
from bn3d.decoders import SweepDecoder3D
from bn3d.bpauli import bcommute
from bn3d.noise import PauliErrorModel
from bn3d.utils import set_where


@pytest.mark.skip(reason='sparse')
class TestSweepDecoder3D:

    @pytest.fixture
    def code(self):
        return Toric3DCode(3, 4, 5)

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
        error.site('Z', (2, 1, 2))
        assert bsf_wt(error.to_bsf()) == 1

        # Measure the syndrome and ensure non-triviality.
        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        correction = decoder.decode(code, syndrome)
        total_error = (error.to_bsf() + correction) % 2
        assert np.all(bcommute(code.stabilizers, total_error) == 0)

    def test_decode_many_Z_errors(self, decoder, code):
        error = Toric3DPauli(code)
        error.site('Z', (1, 0, 0))
        error.site('Z', (0, 1, 0))
        error.site('Z', (0, 0, 3))
        assert bsf_wt(error.to_bsf()) == 3

        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        correction = decoder.decode(code, syndrome)
        total_error = (error.to_bsf() + correction) % 2
        assert np.all(bcommute(code.stabilizers, total_error) == 0)

    def test_unable_to_decode_X_error(self, decoder, code):
        error = Toric3DPauli(code)
        error.site('X', (1, 0, 2))
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
            error.site('Z', site)
            syndrome = code.measure_syndrome(error)
            correction = decoder.decode(code, syndrome)
            total_error = (error.to_bsf() + correction) % 2
            assert np.all(bcommute(code.stabilizers, total_error) == 0)

    def test_decode_error_on_two_edges_sharing_same_vertex(self):
        code = Toric3DCode(3, 3, 3)
        decoder = SweepDecoder3D()
        error_pauli = Toric3DPauli(code)
        error_pauli.site('Z', (1, 0, 0))
        error_pauli.site('Z', (0, 1, 0))
        error = error_pauli.to_bsf()
        syndrome = bcommute(code.stabilizers, error)
        correction = decoder.decode(code, syndrome)
        total_error = (error + correction) % 2
        assert np.all(bcommute(code.stabilizers, total_error) == 0)

    def test_decode_with_general_Z_noise(self):
        code = Toric3DCode(3, 3, 3)
        decoder = SweepDecoder3D()
        np.random.seed(0)
        error_model = PauliErrorModel(0, 0, 1)

        in_codespace = []
        for i in range(100):
            error = error_model.generate(
                code, probability=0.1, rng=np.random
            )
            syndrome = bcommute(code.stabilizers, error)
            correction = decoder.decode(code, syndrome)
            total_error = (error + correction) % 2
            in_codespace.append(
                np.all(bcommute(code.stabilizers, total_error) == 0)
            )

        # Some will just be fails and not in code space, but assert that at
        # least some of them ended up in the code space.
        assert any(in_codespace)

    @pytest.mark.parametrize(
        'edge_location, faces_flipped',
        [
            ((0, 0, 0, 0), {
                (1, 0, 0, 0), (1, 0, 0, 2), (2, 0, 0, 0), (2, 0, 2, 0)
            }),
            ((1, 0, 0, 0), {
                (0, 0, 0, 0), (0, 0, 0, 2), (2, 0, 0, 0), (2, 2, 0, 0)
            }),
            ((2, 0, 0, 0), {
                (0, 0, 0, 0), (0, 0, 2, 0), (1, 0, 0, 0), (1, 2, 0, 0)
            }),
        ]
    )
    def test_flip_edge(self, edge_location, faces_flipped):
        code = Toric3DCode(3, 3, 3)
        decoder = SweepDecoder3D()
        signs = np.zeros(code.shape, dtype=np.uint)
        decoder.flip_edge(edge_location, signs)
        assert set_where(signs) == faces_flipped

    def test_decode_loop_step_by_step(self):
        code = Toric3DCode(3, 3, 3)
        decoder = SweepDecoder3D()

        error_pauli = Toric3DPauli(code)
        sites = [
            (0, 0, 0, 0), (1, 1, 0, 0), (0, 0, 1, 0), (1, 0, 0, 0),
        ]
        for site in sites:
            error_pauli.site('Z', site)
        assert set_where(error_pauli._zs) == set(sites)
        error = error_pauli.to_bsf()

        # Intialize the correction.
        correction = Toric3DPauli(code)

        # Compute the syndrome.
        syndrome = bcommute(code.stabilizers, error)

        signs = np.reshape(
            decoder.get_face_syndromes(code, syndrome),
            newshape=code.shape
        )
        assert np.all(signs.reshape(code.n_k_d[0]) == syndrome[:code.n_k_d[0]])
        assert set_where(signs) == {
            (0, 0, 0, 0), (0, 0, 0, 2), (0, 1, 0, 0), (0, 1, 0, 2),
            (1, 0, 0, 0), (1, 0, 0, 2), (1, 0, 1, 0), (1, 0, 1, 2),
            (2, 0, 1, 0), (2, 0, 2, 0), (2, 1, 0, 0), (2, 2, 0, 0),
        }

        signs = decoder.sweep_move(signs, correction)
        assert set_where(correction._zs) == {
            (0, 0, 1, 0), (1, 1, 0, 0),
            (2, 0, 0, 0), (2, 0, 0, 2),
        }
        assert set_where(signs) == {
            (0, 0, 2, 0), (0, 0, 2, 2),
            (1, 2, 0, 0), (1, 2, 0, 2),
            (2, 2, 0, 0), (2, 0, 2, 0),
        }

        signs = decoder.sweep_move(signs, correction)
        assert set_where(correction._zs) == {
            (0, 0, 1, 0), (1, 1, 0, 0),
            (2, 0, 0, 0), (2, 0, 0, 2),
            (0, 2, 0, 0), (1, 0, 2, 0)
        }
        assert np.all(signs == 0)

        total_error = (error + correction.to_bsf()) % 2
        vertex_operator = Toric3DPauli(code)
        vertex_operator.vertex('Z', (0, 0, 0))
        assert np.all(total_error == vertex_operator.to_bsf())

        assert np.all(bcommute(code.stabilizers, total_error) == 0)

    def test_decode_loop_ok(self):
        code = Toric3DCode(3, 3, 3)
        decoder = SweepDecoder3D()

        error_pauli = Toric3DPauli(code)
        sites = [
            (0, 0, 0, 0), (1, 1, 0, 0), (0, 0, 1, 0), (1, 0, 0, 0)
        ]
        for site in sites:
            error_pauli.site('Z', site)
        assert set_where(error_pauli._zs) == set(sites)
        error = error_pauli.to_bsf()

        # Compute the syndrome.
        syndrome = bcommute(code.stabilizers, error)

        signs = np.reshape(
            decoder.get_face_syndromes(code, syndrome),
            newshape=code.shape
        )
        assert set_where(signs) == {
            (0, 0, 0, 0), (0, 0, 0, 2), (0, 1, 0, 0), (0, 1, 0, 2),
            (1, 0, 0, 0), (1, 0, 0, 2), (1, 0, 1, 0), (1, 0, 1, 2),
            (2, 0, 1, 0), (2, 0, 2, 0), (2, 1, 0, 0), (2, 2, 0, 0)
        }

        assert np.all(signs.reshape(code.n_k_d[0]) == syndrome[:code.n_k_d[0]])

        correction = decoder.decode(code, syndrome)
        total_error = (error + correction) % 2

        assert np.all(bcommute(code.stabilizers, total_error) == 0)

    def test_oscillating_cycle_fail(self):
        code = Toric3DCode(3, 3, 3)
        decoder = SweepDecoder3D()

        error_pauli = Toric3DPauli(code)
        sites = [
            (0, 0, 1, 0), (0, 0, 1, 1), (0, 0, 2, 0), (0, 0, 2, 1),
            (0, 0, 2, 2), (0, 1, 1, 2), (0, 2, 0, 0), (0, 2, 0, 1),
            (0, 2, 0, 2), (1, 1, 0, 1), (1, 1, 2, 0), (1, 1, 2, 2),
            (1, 2, 0, 0), (1, 2, 0, 1), (1, 2, 0, 2), (1, 2, 2, 0),
            (1, 2, 2, 1), (1, 2, 2, 2), (2, 1, 0, 0), (2, 1, 1, 1),
            (2, 1, 2, 1),
        ]
        assert len(set(sites)) == 21
        for site in sites:
            error_pauli.site('Z', site)
        error = error_pauli.to_bsf()

        syndrome = bcommute(code.stabilizers, error)

        # Signs array.
        signs = decoder.get_sign_array(code, syndrome)

        # Keep a copy of the initial signs array.
        start_signs = signs.copy()

        # Keep track of the correction to apply.
        correction = Toric3DPauli(code)

        # Sweep 3 times.
        for i_sweep in range(3):
            signs = decoder.sweep_move(signs, correction)

        # Back to the start again.
        assert np.all(signs == start_signs)

        # The total correction is trivial.
        assert np.all(bcommute(code.stabilizers, correction.to_bsf()) == 0)

        # The total error still is not in code space.
        total_error = (error + correction.to_bsf()) % 2
        assert np.any(bcommute(code.stabilizers, total_error) != 0)

    def test_never_ending_staircase_fails(self):
        code = Toric3DCode(3, 3, 3)
        decoder = SweepDecoder3D()

        # Weight-8 Z error that may start infinite loop in sweep decoder.
        error_pauli = Toric3DPauli(code)
        sites = [
            (0, 0, 2, 2), (0, 1, 1, 1), (0, 2, 0, 2), (1, 0, 0, 0),
            (1, 1, 0, 2), (1, 2, 2, 1), (2, 1, 2, 1), (2, 2, 0, 0)
        ]
        for site in sites:
            error_pauli.site('Z', site)
        error = error_pauli.to_bsf()
        # assert error.sum() == 8

        # Compute the syndrome and make sure it's nontrivial.
        syndrome = bcommute(code.stabilizers, error)
        assert np.any(syndrome)

        # Check face X stabilizer syndrome measurements.
        expected_syndrome_faces = [
            (0, 0, 0, 0), (0, 0, 0, 2), (0, 1, 0, 1), (0, 1, 0, 2),
            (0, 1, 1, 1), (0, 1, 2, 1), (0, 2, 0, 0), (0, 2, 2, 1),
            (1, 0, 2, 2), (1, 1, 0, 0), (1, 1, 1, 0), (1, 1, 1, 1),
            (1, 1, 2, 1), (1, 2, 0, 0), (1, 2, 0, 1), (1, 2, 0, 2),
            (2, 0, 0, 0), (2, 0, 0, 2), (2, 0, 1, 2), (2, 0, 2, 2),
            (2, 1, 0, 1), (2, 1, 0, 2), (2, 1, 1, 1), (2, 1, 2, 1),
            (2, 2, 0, 0), (2, 2, 0, 2), (2, 2, 2, 1), (2, 2, 2, 2)
        ]
        assert np.all(
            np.array(expected_syndrome_faces).T
            == np.where(syndrome[:code.n_k_d[0]].reshape(3, 3, 3, 3))
        )

        # Attempt to perform decoding.
        correction = decoder.decode(code, syndrome)

        total_error = (error + correction) % 2

        # Assert that decoding has failed.
        np.any(bcommute(code.stabilizers, total_error))

    def test_sweep_move_two_edges(self):
        code = Toric3DCode(3, 3, 3)
        decoder = SweepDecoder3D()

        error = Toric3DPauli(code)
        error.site('Z', (0, 1, 0))
        error.site('Z', (1, 0, 0))

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
        signs_1 = decoder.sweep_move(signs, correction)
        assert np.all(expected_signs_1 == signs_1)

        # Expected signs after two sweeps, should be all gone.
        signs_2 = decoder.sweep_move(signs_1, correction)
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
