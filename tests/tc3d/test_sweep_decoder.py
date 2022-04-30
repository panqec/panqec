import itertools
import pytest
import numpy as np
from panqec.codes import Toric3DCode
from panqec.decoders import SweepDecoder3D
from panqec.bpauli import bcommute, bsf_wt
from panqec.error_models import PauliErrorModel
from panqec.utils import dict_where, set_where, edge_coords, face_coords


@pytest.mark.skip(reason='refactor')
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
        syndrome = np.zeros(shape=len(code.stabilizer_matrix), dtype=np.uint)
        correction = decoder.decode(code, syndrome)
        assert correction.shape == 2*code.n
        assert np.all(bcommute(code.stabilizer_matrix, correction) == 0)
        assert issubclass(correction.dtype.type, np.integer)

    def test_decode_Z_error(self, decoder, code):
        error = dict()
        error[(2, 1, 2)] = 'Z'
        assert bsf_wt(code.to_bsf(error)) == 1

        # Measure the syndrome and ensure non-triviality.
        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        correction = decoder.decode(code, syndrome)
        total_error = (code.to_bsf(error) + correction) % 2
        assert np.all(bcommute(code.stabilizer_matrix, total_error) == 0)

    def test_decode_many_Z_errors(self, decoder, code):
        error = dict()
        error[(1, 0, 0)] = 'Z'
        error[(0, 1, 0)] = 'Z'
        error[(0, 0, 3)] = 'Z'
        assert bsf_wt(code.to_bsf(error)) == 3

        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        correction = decoder.decode(code, syndrome)
        total_error = (code.to_bsf(error) + correction) % 2
        assert np.all(bcommute(code.stabilizer_matrix, total_error) == 0)

    def test_unable_to_decode_X_error(self, decoder, code):
        error = dict()
        error[(1, 0, 2)] = 'X'
        assert bsf_wt(code.to_bsf(error)) == 1

        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        correction = decoder.decode(code, syndrome)
        assert np.all(correction.todense() == 0)

        total_error = (code.to_bsf(error) + correction) % 2
        assert np.all(code.to_bsf(error) == total_error)

        assert np.any(bcommute(code.stabilizer_matrix, total_error) != 0)

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
            error = dict()
            error[site] = 'Z'
            syndrome = code.measure_syndrome(error)
            correction = decoder.decode(code, syndrome)
            total_error = (code.to_bsf(error) + correction) % 2
            assert np.all(bcommute(code.stabilizer_matrix, total_error) == 0)

    def test_decode_error_on_two_edges_sharing_same_vertex(self):
        code = Toric3DCode(3, 3, 3)
        decoder = SweepDecoder3D()
        error_pauli = dict()
        error_pauli[(1, 0, 0)] = 'Z'
        error_pauli[(0, 1, 0)] = 'Z'
        error = code.to_bsf(error_pauli)
        syndrome = bcommute(code.stabilizer_matrix, error)
        correction = decoder.decode(code, syndrome)
        total_error = (error + correction) % 2
        assert np.all(bcommute(code.stabilizer_matrix, total_error) == 0)

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
            syndrome = bcommute(code.stabilizer_matrix, error)
            correction = decoder.decode(code, syndrome)
            total_error = (error + correction) % 2
            in_codespace.append(
                np.all(bcommute(code.stabilizer_matrix, total_error) == 0)
            )

        # Some will just be fails and not in code space, but assert that at
        # least some of them ended up in the code space.
        assert any(in_codespace)

    @pytest.mark.parametrize(
        'edge_location, faces_flipped',
        [
            ((1, 0, 0), {
                (1, 1, 0), (1, 5, 0), (1, 0, 1), (1, 0, 5)
            }),
            ((0, 1, 0), {
                (1, 1, 0), (5, 1, 0), (0, 1, 1), (0, 1, 5)
            }),
            ((0, 0, 1), {
                (1, 0, 1), (5, 0, 1), (0, 1, 1), (0, 5, 1)
            }),
        ]
    )
    def test_flip_edge(self, edge_location, faces_flipped):
        code = Toric3DCode(3, 3, 3)
        decoder = SweepDecoder3D()
        signs = decoder.get_initial_state(
            code, np.zeros(code.stabilizers.shape[0])
        )
        decoder.flip_edge(edge_location, signs, code)
        assert dict_where(signs) == faces_flipped

    def test_decode_loop_step_by_step(self):
        code = Toric3DCode(3, 3, 3)
        decoder = SweepDecoder3D()

        error_pauli = dict()
        sites = [
            (0, 0, 0, 0), (1, 1, 0, 0), (0, 0, 1, 0), (1, 0, 0, 0),
        ]
        for site in sites:
            error_pauli[site] = 'Z'
        assert set_where(error_pauli._zs) == set(sites)
        error = code.to_bsf(error_pauli)

        # Intialize the correction.
        correction = dict()

        # Compute the syndrome.
        syndrome = bcommute(code.stabilizer_matrix, error)

        signs = decoder.get_initial_state(code, syndrome)
        assert np.all(
            rebuild_syndrome(code, signs)[:code.n_k_d[0]]
            == syndrome[:code.n_k_d[0]]
        )
        assert np.all(signs.reshape(code.n) == syndrome[:code.n])
        assert set_where(signs) == {
            (0, 0, 0, 0), (0, 0, 0, 2), (0, 1, 0, 0), (0, 1, 0, 2),
            (1, 0, 0, 0), (1, 0, 0, 2), (1, 0, 1, 0), (1, 0, 1, 2),
            (2, 0, 1, 0), (2, 0, 2, 0), (2, 1, 0, 0), (2, 2, 0, 0),
        }

        signs = decoder.sweep_move(signs, correction, code)
        assert find_sites(correction) == set(edge_coords([
            (0, 0, 1, 0), (1, 1, 0, 0),
            (2, 0, 0, 0), (2, 0, 0, 2),
        ], code.size))
        assert dict_where(signs) == set(face_coords([
            (0, 0, 2, 0), (0, 0, 2, 2),
            (1, 2, 0, 0), (1, 2, 0, 2),
            (2, 2, 0, 0), (2, 0, 2, 0),
        ], code.size))

        signs = decoder.sweep_move(signs, correction, code)
        assert find_sites(correction) == set(edge_coords([
            (0, 0, 1, 0), (1, 1, 0, 0),
            (2, 0, 0, 0), (2, 0, 0, 2),
            (0, 2, 0, 0), (1, 0, 2, 0)
        ], code.size))
        assert np.all(np.array(list(signs.values())) == 0)

        total_error = (error + code.to_bsf(correction)) % 2
        vertex_operator = code.get_stabilizer((0, 0, 0))
        assert np.all(total_error == code.to_bsf(vertex_operator))

        assert np.all(bcommute(code.stabilizer_matrix, total_error) == 0)

    def test_decode_loop_ok(self):
        code = Toric3DCode(3, 3, 3)
        decoder = SweepDecoder3D()

        error_pauli = dict()
        """
        sites = [
            (0, 0, 0, 0), (1, 1, 0, 0), (0, 0, 1, 0), (1, 0, 0, 0)
        ]
        """
        sites = [(1, 0, 0), (2, 1, 0), (1, 2, 0), (0, 1, 0)]
        for site in sites:
            error_pauli[site] = 'Z'
        assert set_where(error_pauli._zs) == set(sites)
        error = code.to_bsf(error_pauli)

        # Compute the syndrome.
        syndrome = bcommute(code.stabilizer_matrix, error)

        signs = decoder.get_initial_state(code, syndrome)
        """
        assert dict_where(signs) == {
            (0, 0, 0, 0), (0, 0, 0, 2), (0, 1, 0, 0), (0, 1, 0, 2),
            (1, 0, 0, 0), (1, 0, 0, 2), (1, 0, 1, 0), (1, 0, 1, 2),
            (2, 0, 1, 0), (2, 0, 2, 0), (2, 1, 0, 0), (2, 2, 0, 0)
        }
        """
        assert dict_where(signs) == {
            (1, 2, 1), (2, 1, 1), (1, 0, 1), (3, 1, 0), (1, 5, 0), (1, 0, 5),
            (0, 1, 5), (1, 2, 5), (2, 1, 5), (5, 1, 0), (1, 3, 0), (0, 1, 1)
        }

        reconstructed_syndrome = rebuild_syndrome(code, signs)
        assert np.all(
            reconstructed_syndrome[:code.n_k_d[0]] == syndrome[:code.n_k_d[0]]
        )

        correction = decoder.decode(code, syndrome)
        total_error = (error.todense() + correction) % 2

        assert np.all(bcommute(code.stabilizer_matrix, total_error) == 0)

    def test_oscillating_cycle_fail(self):
        code = Toric3DCode(3, 3, 3)
        decoder = SweepDecoder3D()

        error_pauli = dict()
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
            error_pauli[site] = 'Z'
        error = code.to_bsf(error_pauli)

        syndrome = bcommute(code.stabilizer_matrix, error)

        # Signs array.
        signs = decoder.get_initial_state(code, syndrome)

        # Keep a copy of the initial signs array.
        start_signs = signs.copy()

        # Keep track of the correction to apply.
        correction = dict()

        # Sweep 3 times.
        for i_sweep in range(3):
            signs = decoder.sweep_move(signs, correction, code)

        # Back to the start again.
        assert np.all(signs == start_signs)

        # The total correction is trivial.
        assert np.all(bcommute(code.stabilizer_matrix, code.to_bsf(correction)) == 0)

        # The total error still is not in code space.
        total_error = (error + code.to_bsf(correction)) % 2
        assert np.any(bcommute(code.stabilizer_matrix, total_error) != 0)

    def test_never_ending_staircase_fails(self):
        code = Toric3DCode(3, 3, 3)
        decoder = SweepDecoder3D()

        # Weight-8 Z error that may start infinite loop in sweep decoder.
        error_pauli = dict()
        sites = edge_coords([
            (0, 0, 2, 2), (0, 1, 1, 1), (0, 2, 0, 2), (1, 0, 0, 0),
            (1, 1, 0, 2), (1, 2, 2, 1), (2, 1, 2, 1), (2, 2, 0, 0)
        ], code.size)
        for site in sites:
            error_pauli[site] = 'Z'
        error = code.to_bsf(error_pauli)
        # assert error.sum() == 8

        # Compute the syndrome and make sure it's nontrivial.
        syndrome = bcommute(code.stabilizer_matrix, error)
        assert np.any(syndrome)

        # Check face X stabilizer syndrome measurements.
        expected_syndrome_faces = face_coords([
            (0, 0, 0, 0), (0, 0, 0, 2), (0, 1, 0, 1), (0, 1, 0, 2),
            (0, 1, 1, 1), (0, 1, 2, 1), (0, 2, 0, 0), (0, 2, 2, 1),
            (1, 0, 2, 2), (1, 1, 0, 0), (1, 1, 1, 0), (1, 1, 1, 1),
            (1, 1, 2, 1), (1, 2, 0, 0), (1, 2, 0, 1), (1, 2, 0, 2),
            (2, 0, 0, 0), (2, 0, 0, 2), (2, 0, 1, 2), (2, 0, 2, 2),
            (2, 1, 0, 1), (2, 1, 0, 2), (2, 1, 1, 1), (2, 1, 2, 1),
            (2, 2, 0, 0), (2, 2, 0, 2), (2, 2, 2, 1), (2, 2, 2, 2)
        ], code.size)
        expected_signs = {k: 0 for k in code.face_index}
        for k in expected_syndrome_faces:
            expected_signs[k] = 1
        expected_syndrome = rebuild_syndrome(
            code, expected_signs
        )
        assert np.all(syndrome == expected_syndrome)
        """
        assert np.all(
            np.array(expected_syndrome_faces).T
            == np.where(syndrome[:code.n].reshape(3, 3, 3, 3))
        )
        """

        # Attempt to perform decoding.
        correction = decoder.decode(code, syndrome)

        total_error = (error + correction.todense()) % 2

        # Assert that decoding has failed.
        np.any(bcommute(code.stabilizer_matrix, total_error))

    def test_sweep_move_two_edges(self):
        code = Toric3DCode(3, 3, 3)
        decoder = SweepDecoder3D()

        error = dict()
        error[(0, 1, 0)] = 'Z'
        error[(1, 0, 0)] = 'Z'

        syndrome = bcommute(code.stabilizer_matrix, code.to_bsf(error))

        correction = dict()

        # Syndrome from errors on x edge and y edge on vertex (0, 0, 0).
        signs = {k: 0 for k in code.face_index}
        faces = [
            (1, 5, 0), (0, 1, 1), (0, 1, 5),
            (5, 1, 0), (1, 0, 1), (1, 0, 5),
        ]
        for face in faces:
            signs[face] = 1

        assert decoder.get_initial_state(code, syndrome) == signs

        # Expected signs after one sweep.
        expected_faces_1 = [
            (0, 5, 1), (0, 5, 5),
            (1, 5, 0),
            (5, 0, 1), (5, 0, 5),
            (5, 1, 0),
        ]
        signs_1 = decoder.sweep_move(signs, correction, code)
        faces_1 = [k for k, v in signs_1.items() if v]
        assert set(expected_faces_1) == set(faces_1)

        # Expected signs after two sweeps, should be all gone.
        signs_2 = decoder.sweep_move(signs_1, correction, code)
        assert all(np.array(list(signs_2.values())) == 0)

        expected_correction = {
            (0, 5, 0): 'Z',
            (5, 0, 0): 'Z',
            (0, 0, 1): 'Z',
            (0, 0, 5): 'Z',
        }

        assert not np.any((
            correction.to_bsf() != expected_correction.to_bsf()
        ).toarray())


def find_sites(error_pauli):
    """List of sites where Pauli has support over."""
    return set([
        location
        for location, index in error_pauli.code.qubit_index.items()
        if index in np.where(error_pauli._zs.toarray()[0])[0]
    ])


def rebuild_syndrome(code, signs):
    reconstructed_syndrome = np.zeros(code.stabilizers.shape[0], dtype=np.uint)
    for location, index in code.face_index.items():
        if signs[location]:
            reconstructed_syndrome[index] = 1
    return reconstructed_syndrome
