import itertools
import pytest
import numpy as np
from panqec.codes import Toric3DCode
from panqec.decoders import SweepDecoder3D
from panqec.bpauli import bsf_wt
from panqec.error_models import PauliErrorModel
from panqec.utils import edge_coords, face_coords
from tests.decoders.decoder_test import DecoderTest


class TestSweepDecoder3D(DecoderTest):

    @pytest.fixture
    def code(self):
        return Toric3DCode(3, 4, 5)

    @pytest.fixture
    def error_model(self):
        return PauliErrorModel(0, 0, 1)

    @pytest.fixture
    def decoder(self, code, error_model):
        error_rate = 0.5
        return SweepDecoder3D(code, error_model, error_rate)

    @pytest.fixture
    def allowed_paulis(self):
        return ['Z']

    def test_decode_Z_error(self, decoder, code):
        error = code.to_bsf({
            (2, 1, 2): 'Z',
        })
        assert bsf_wt(error) == 1

        # Measure the syndrome and ensure non-triviality.
        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        correction = decoder.decode(syndrome)
        total_error = (error + correction) % 2
        assert np.all(code.measure_syndrome(total_error) == 0)

    def test_decode_many_Z_errors(self, decoder, code):
        error = dict()
        error = code.to_bsf({
            (1, 0, 0): 'Z',
            (0, 1, 0): 'Z',
            (0, 0, 3): 'Z',
        })
        assert bsf_wt(error) == 3

        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        correction = decoder.decode(syndrome)
        total_error = (error + correction) % 2
        assert np.all(code.measure_syndrome(total_error) == 0)

    def test_unable_to_decode_X_error(self, decoder, code):
        error = code.to_bsf({
            (1, 0, 2): 'X'
        })
        assert bsf_wt(error) == 1

        syndrome = code.measure_syndrome(error)
        assert np.any(syndrome != 0)

        correction = decoder.decode(syndrome)
        assert np.all(correction == 0)

        total_error = (error + correction) % 2
        assert np.all(error == total_error)

        assert np.any(code.measure_syndrome(total_error) != 0)

    def test_decode_many_codes_and_errors_with_same_decoder(self):

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

        error_model = PauliErrorModel(0, 0, 1)
        error_rate = 0.5

        for code, site in itertools.product(codes, sites):
            decoder = SweepDecoder3D(code, error_model, error_rate)
            error = code.to_bsf({
                site: 'Z',
            })
            syndrome = code.measure_syndrome(error)
            correction = decoder.decode(syndrome)
            total_error = (error + correction) % 2
            assert np.all(code.measure_syndrome(total_error) == 0)


class Test3x3x3SweepDecoder3D(DecoderTest):

    @pytest.fixture
    def code(self):
        return Toric3DCode(3, 3, 3)

    @pytest.fixture
    def decoder(self, code, error_model):
        error_rate = 0.1
        return SweepDecoder3D(code, error_model, error_rate)

    @pytest.fixture
    def error_model(self):
        return PauliErrorModel(0, 0, 1)

    @pytest.fixture
    def allowed_paulis(self):
        return ['Z']

    def test_decode_error_on_two_edges_sharing_same_vertex(
        self, code, decoder
    ):
        error = code.to_bsf({
            (1, 0, 0): 'Z',
            (0, 1, 0): 'Z',
        })
        syndrome = code.measure_syndrome(error)
        correction = decoder.decode(syndrome)
        total_error = (error + correction) % 2
        assert np.all(code.measure_syndrome(total_error) == 0)

    def test_decode_with_general_Z_noise(self):
        code = Toric3DCode(3, 3, 3)
        error_model = PauliErrorModel(0, 0, 1)
        error_rate = 0.1
        decoder = SweepDecoder3D(code, error_model, error_rate)
        np.random.seed(0)

        in_codespace = []
        for i in range(100):
            error = error_model.generate(
                code, error_rate=error_rate, rng=np.random
            )
            syndrome = code.measure_syndrome(error)
            correction = decoder.decode(syndrome)
            total_error = (error + correction) % 2
            in_codespace.append(
                np.all(code.measure_syndrome(total_error) == 0)
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
    def test_flip_edge(self, edge_location, faces_flipped, code, decoder):
        signs = decoder.get_initial_state(
            np.zeros(code.stabilizer_matrix.shape[0])
        )
        decoder.flip_edge(edge_location, signs)
        assert faces_flipped == {
            code.stabilizer_coordinates[index]
            for index in np.where(signs)[0]
        }

    def test_decode_loop_step_by_step(self, code, decoder):
        sites = set(edge_coords([
            (0, 0, 0, 0), (1, 1, 0, 0), (0, 0, 1, 0), (1, 0, 0, 0),
        ], code.size))

        error_pauli = dict()
        for site in sites:
            error_pauli[site] = 'Z'
        assert set(error_pauli.keys()) == set(sites)
        error = code.to_bsf(error_pauli)

        # Initialize the correction.
        correction = dict()

        # Compute the syndrome.
        syndrome = code.measure_syndrome(error)

        signs = decoder.get_initial_state(syndrome)
        assert np.all(signs == syndrome)
        # assert np.all(
        #     rebuild_syndrome(code, signs)[:code.n_k_d[0]]
        #     == syndrome[:code.n_k_d[0]]
        # )

        # assert np.all(signs.reshape(code.n) == syndrome[:code.n])
        assert {
            code.stabilizer_coordinates[index]
            for index in np.where(signs)[0]
        } == set(face_coords([
            (0, 0, 0, 0), (0, 0, 0, 2), (0, 1, 0, 0), (0, 1, 0, 2),
            (1, 0, 0, 0), (1, 0, 0, 2), (1, 0, 1, 0), (1, 0, 1, 2),
            (2, 0, 1, 0), (2, 0, 2, 0), (2, 1, 0, 0), (2, 2, 0, 0),
        ], code.size))

        signs = decoder.sweep_move(signs, correction)
        assert set(correction.keys()) == set(edge_coords([
            (0, 0, 1, 0), (1, 1, 0, 0),
            (2, 0, 0, 0), (2, 0, 0, 2),
        ], code.size))
        assert {
            code.stabilizer_coordinates[index]
            for index in np.where(signs)[0]
        } == set(face_coords([
            (0, 0, 2, 0), (0, 0, 2, 2),
            (1, 2, 0, 0), (1, 2, 0, 2),
            (2, 2, 0, 0), (2, 0, 2, 0),
        ], code.size))

        signs = decoder.sweep_move(signs, correction)
        assert set(correction.keys()) == set(edge_coords([
            (0, 0, 1, 0), (1, 1, 0, 0),
            (2, 0, 0, 0), (2, 0, 0, 2),
            (0, 2, 0, 0), (1, 0, 2, 0)
        ], code.size))
        assert np.all(signs == 0)

        total_error = (error + code.to_bsf(correction)) % 2
        vertex_operator = code.get_stabilizer((0, 0, 0))
        assert np.all(total_error == code.to_bsf(vertex_operator))

        assert np.all(code.measure_syndrome(total_error) == 0)

    def test_decode_loop_ok(self, code, decoder):

        error_pauli = dict()
        """
        sites = [
            (0, 0, 0, 0), (1, 1, 0, 0), (0, 0, 1, 0), (1, 0, 0, 0)
        ]
        """
        sites = [(1, 0, 0), (2, 1, 0), (1, 2, 0), (0, 1, 0)]
        for site in sites:
            error_pauli[site] = 'Z'
        assert set(error_pauli.keys()) == set(sites)
        error = code.to_bsf(error_pauli)

        # Compute the syndrome.
        syndrome = code.measure_syndrome(error)

        signs = decoder.get_initial_state(syndrome)
        """
        assert dict_where(signs) == {
            (0, 0, 0, 0), (0, 0, 0, 2), (0, 1, 0, 0), (0, 1, 0, 2),
            (1, 0, 0, 0), (1, 0, 0, 2), (1, 0, 1, 0), (1, 0, 1, 2),
            (2, 0, 1, 0), (2, 0, 2, 0), (2, 1, 0, 0), (2, 2, 0, 0)
        }
        """
        assert {
            code.stabilizer_coordinates[index]
            for index in np.where(signs)[0]
        } == {
            (1, 2, 1), (2, 1, 1), (1, 0, 1), (3, 1, 0), (1, 5, 0), (1, 0, 5),
            (0, 1, 5), (1, 2, 5), (2, 1, 5), (5, 1, 0), (1, 3, 0), (0, 1, 1)
        }

        assert np.all(signs == syndrome)

        correction = decoder.decode(syndrome)
        total_error = (error + correction) % 2

        assert np.all(code.measure_syndrome(total_error) == 0)

    def test_oscillating_cycle_fail(self, code, decoder):

        sites = edge_coords([
            (0, 0, 1, 0), (0, 0, 1, 1), (0, 0, 2, 0), (0, 0, 2, 1),
            (0, 0, 2, 2), (0, 1, 1, 2), (0, 2, 0, 0), (0, 2, 0, 1),
            (0, 2, 0, 2), (1, 1, 0, 1), (1, 1, 2, 0), (1, 1, 2, 2),
            (1, 2, 0, 0), (1, 2, 0, 1), (1, 2, 0, 2), (1, 2, 2, 0),
            (1, 2, 2, 1), (1, 2, 2, 2), (2, 1, 0, 0), (2, 1, 1, 1),
            (2, 1, 2, 1),
        ], code.size)
        assert len(set(sites)) == 21

        error_pauli = dict()
        for site in sites:
            error_pauli[site] = 'Z'
        error = code.to_bsf(error_pauli)

        syndrome = code.measure_syndrome(error)

        # Signs array.
        signs = decoder.get_initial_state(syndrome)

        # Keep a copy of the initial signs array.
        start_signs = signs.copy()

        # Keep track of the correction to apply.
        correction = dict()

        # Sweep 3 times.
        for i_sweep in range(3):
            signs = decoder.sweep_move(signs, correction)

        # Back to the start again.
        assert np.all(signs == start_signs)

        # The total correction is trivial.
        assert np.all(
            code.measure_syndrome(code.to_bsf(correction)) == 0
        )

        # The total error still is not in code space.
        total_error = (error + code.to_bsf(correction)) % 2
        assert np.any(code.measure_syndrome(total_error) != 0)

    def test_never_ending_staircase_fails(self, code, decoder):

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
        syndrome = code.measure_syndrome(error)
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
        expected_signs = {k: 0 for k in code.type_index('face')}
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
        correction = decoder.decode(syndrome)

        total_error = (error + correction) % 2

        # Assert that decoding has failed.
        np.any(code.measure_syndrome(total_error))

    def test_sweep_move_two_edges(self, code, decoder):

        error = {
            (0, 1, 0): 'Z',
            (1, 0, 0): 'Z',
        }

        syndrome = code.measure_syndrome(code.to_bsf(error))

        correction = dict()

        # Syndrome from errors on x edge and y edge on vertex (0, 0, 0).
        signs = {k: 0 for k in code.type_index('face')}
        faces = [
            (1, 5, 0), (0, 1, 1), (0, 1, 5),
            (5, 1, 0), (1, 0, 1), (1, 0, 5),
        ]
        signs = np.zeros(code.stabilizer_matrix.shape[0], dtype=np.uint)
        for face in faces:
            signs[code.stabilizer_index[face]] = 1

        assert np.all(decoder.get_initial_state(syndrome) == signs)

        # Expected signs after one sweep.
        expected_faces_1 = [
            (0, 5, 1), (0, 5, 5),
            (1, 5, 0),
            (5, 0, 1), (5, 0, 5),
            (5, 1, 0),
        ]
        signs_1 = decoder.sweep_move(signs, correction)
        faces_1 = {
            code.stabilizer_coordinates[index]
            for index in np.where(signs_1)[0]
        }
        assert set(expected_faces_1) == set(faces_1)

        # Expected signs after two sweeps, should be all gone.
        signs_2 = decoder.sweep_move(signs_1, correction)
        assert all(signs_2 == 0)

        expected_correction = {
            (0, 5, 0): 'Z',
            (5, 0, 0): 'Z',
            (0, 0, 1): 'Z',
            (0, 0, 5): 'Z',
        }

        assert correction == expected_correction


def find_sites(error_pauli):
    """List of sites where Pauli has support over."""
    return set([
        location
        for location, index in error_pauli.code.qubit_index.items()
        if index in np.where(error_pauli._zs.toarray()[0])[0]
    ])


def rebuild_syndrome(code, signs):
    reconstructed_syndrome = np.zeros(
        code.stabilizer_matrix.shape[0], dtype=np.uint
    )
    for location, index in code.type_index('face').items():
        if signs[location]:
            reconstructed_syndrome[index] = 1
    return reconstructed_syndrome
