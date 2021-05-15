import numpy as np
import itertools
import pytest
from qecsim.paulitools import bsf_to_pauli, bsf_wt
from bn3d.noise import (
    generate_pauli_noise, deform_operator, get_deformed_weights
)
from bn3d.bpauli import get_bvector_index
from bn3d.noise import PauliErrorModel, XNoiseOnYZEdgesOnly
from bn3d.tc3d import ToricCode3D, Toric3DPauli


class TestPauliNoise:

    @pytest.fixture(autouse=True)
    def seed_random(self):
        np.random.seed(0)

    @pytest.fixture
    def code(self):
        return ToricCode3D(3, 4, 5)

    @pytest.fixture
    def error_model(self):
        return PauliErrorModel(0.2, 0.3, 0.5)

    def test_label(self, error_model):
        assert error_model.label == 'Pauli X0.2Y0.3Z0.5'

    def test_generate(self, code, error_model):
        probability = 0.1
        error = error_model.generate(code, probability, rng=np.random)
        assert np.any(error != 0), 'Error should be non-trivial'
        assert error.shape == (2*code.n_k_d[0], ), 'Shape incorrect'

    def test_probability_zero(self, code, error_model):
        probability = 0
        error = error_model.generate(code, probability, rng=np.random)
        assert np.all(error == 0), 'Should have no error'

    def test_probability_one(self, code, error_model):
        probability = 1
        error = error_model.generate(code, probability, rng=np.random)

        # Error everywhere so weight is number of qubits.
        assert bsf_wt(error) == code.n_k_d[0], 'Should be error everywhere'

    def test_generate_all_X_errors(self, code):
        probability = 1
        direction = (1, 0, 0)
        error_model = PauliErrorModel(*direction)
        error = error_model.generate(code, probability, rng=np.random)
        assert bsf_to_pauli(error) == 'X'*code.n_k_d[0], (
            'Should be X error everywhere'
        )

    def test_generate_all_Y_errors(self, code):
        probability = 1
        direction = (0, 1, 0)
        error_model = PauliErrorModel(*direction)
        error = error_model.generate(code, probability, rng=np.random)
        assert bsf_to_pauli(error) == 'Y'*code.n_k_d[0], (
            'Should be Y error everywhere'
        )

    def test_generate_all_Z_errors(self, code):
        probability = 1
        direction = (0, 0, 1)
        error_model = PauliErrorModel(*direction)
        error = error_model.generate(code, probability, rng=np.random)
        assert bsf_to_pauli(error) == 'Z'*code.n_k_d[0], (
            'Should be Z error everywhere'
        )

    def test_raise_error_if_direction_does_not_sum_to_1(self):
        with pytest.raises(ValueError):
            PauliErrorModel(0, 0, 0)


class TestGeneratePauliNoise:

    @pytest.fixture(autouse=True)
    def seed_random(self):
        np.random.seed(0)

    def test_generate_pauli_noise(self):
        p_X = 0.1
        p_Y = 0.2
        p_Z = 0.5
        L = 10
        noise = generate_pauli_noise(p_X, p_Y, p_Z, L)
        assert list(noise.shape) == [2*3*L**3]
        assert noise.dtype == np.uint

    def test_no_errors_if_p_zero(self):
        L = 10
        noise = generate_pauli_noise(0, 0, 0, L)
        assert np.all(noise == 0)

    def test_all_X_if_p_X_one(self):
        L = 10
        noise = generate_pauli_noise(1, 0, 0, L)
        assert np.all(noise[:3*L**3] == 1)
        assert np.all(noise[3*L**3:] == 0)

    def test_all_Y_if_p_Y_one(self):
        L = 10
        noise = generate_pauli_noise(0, 1, 0, L)
        assert np.all(noise[:3*L**3] == 1)
        assert np.all(noise[3*L**3:] == 1)

    def test_all_Z_if_p_Z_one(self):
        L = 10
        noise = generate_pauli_noise(0, 0, 1, L)
        assert np.all(noise[:3*L**3] == 0)
        assert np.all(noise[3*L**3:] == 1)

    def test_only_X_if_p_X_only(self):
        L = 10
        noise = generate_pauli_noise(0.5, 0, 0, L)
        assert np.any(noise[:3*L**3] == 1)
        assert np.all(noise[3*L**3:] == 0)

    def test_only_Y_if_p_Y_only(self):
        L = 10
        noise = generate_pauli_noise(0, 0.5, 0, L)
        assert np.all(noise[:3*L**3] == noise[3*L**3:])

    def test_only_Z_if_p_Z_only(self):
        L = 10
        noise = generate_pauli_noise(0, 0, 0.5, L)
        assert np.all(noise[:3*L**3] == 0)
        assert np.any(noise[3*L**3:] == 1)


class TestDeformOperator:

    @pytest.fixture(autouse=True)
    def noise_config(self):
        self.L = 10
        self.noise = generate_pauli_noise(0.1, 0.2, 0.3, self.L)
        self.deformed = deform_operator(self.noise, self.L)

    def test_deform_operator_shape(self):
        assert self.deformed.dtype == np.uint
        assert list(self.deformed.shape) == list(self.noise.shape)

    def test_deformed_is_different(self):
        assert np.any(self.noise != self.deformed)

    def test_deformed_composed_original_has_Ys_only(self):
        composed = (self.deformed + self.noise) % 2
        assert np.all(composed[3*self.L**3:] == composed[:3*self.L**3])

    def test_only_x_edges_are_different(self):
        differing_locations = []
        for edge in range(3):
            for x in range(self.L):
                for y in range(self.L):
                    for z in range(self.L):
                        i_X = get_bvector_index(edge, x, y, z, 0, self.L)
                        i_Z = get_bvector_index(edge, x, y, z, 1, self.L)
                        if (
                            self.deformed[i_X] != self.noise[i_X]
                            or
                            self.deformed[i_Z] != self.noise[i_Z]
                        ):
                            differing_locations.append(
                                (edge, x, y, z)
                            )

        assert len(differing_locations) > 0
        for location in differing_locations:
            edge = location[0]
            assert edge == 0


class TestGetDeformedWeights:

    def test_equal_rates_then_equal_weights(self):
        L = 10
        p_X, p_Y, p_Z = 0.1, 0.1, 0.1
        weights = get_deformed_weights(p_X, p_Y, p_Z, L)
        assert np.all(weights == weights[0])

    def test_zero_error_rate_no_nan(self):
        L = 10
        p_X, p_Y, p_Z = 0, 0, 0
        weights = get_deformed_weights(p_X, p_Y, p_Z, L)
        assert np.all(weights != 0)
        assert np.all(~np.isnan(weights))

    def test_one_error_rate_no_nan(self):
        L = 10
        p_X, p_Y, p_Z = 1, 0, 0
        weights = get_deformed_weights(p_X, p_Y, p_Z, L)
        assert np.all(weights != 0)
        assert np.all(~np.isnan(weights))

    def test_biased_Z_noise_different_weights(self):
        L = 10
        p_X, p_Y, p_Z = 0.5, 0, 0
        weights = get_deformed_weights(p_X, p_Y, p_Z, L)
        assert np.any(weights != weights[0])

    def test_only_x_edges_different_weights(self):
        L = 10
        p_X, p_Y, p_Z = 0.5, 0.1, 0
        weights = get_deformed_weights(p_X, p_Y, p_Z, L)
        assert np.any(weights != weights[0])
        x_edge_indices = [
            get_bvector_index(0, x, y, z, 0, L)
            for x, y, z in itertools.product(range(L), repeat=3)
        ]
        y_edge_indices = [
            get_bvector_index(1, x, y, z, 0, L)
            for x, y, z in itertools.product(range(L), repeat=3)
        ]
        z_edge_indices = [
            get_bvector_index(2, x, y, z, 0, L)
            for x, y, z in itertools.product(range(L), repeat=3)
        ]

        # Weights for the same wedge type should be equal.
        assert np.all(weights[x_edge_indices] == weights[x_edge_indices[0]])
        assert np.all(weights[y_edge_indices] == weights[y_edge_indices[0]])
        assert np.all(weights[z_edge_indices] == weights[z_edge_indices[0]])

        # Weights for y-edges and z-edges should be equal.
        assert np.all(weights[y_edge_indices] == weights[z_edge_indices])

        # Weights for x-edges and z-edges should not be equal.
        assert np.any(weights[x_edge_indices] != weights[z_edge_indices])


class TestXNoiseOnYZEdgesOnly:

    @pytest.fixture(autouse=True)
    def rng(self):
        return np.random.default_rng(seed=0)

    @pytest.fixture
    def code(self):
        return ToricCode3D(3, 4, 5)

    @pytest.fixture
    def error_model(self):
        return XNoiseOnYZEdgesOnly()

    def test_label(self, error_model):
        assert error_model.label == 'X on yz edges'

    def test_generate_zero_probability(self, code, error_model, rng):
        error = error_model.generate(code, probability=0, rng=rng)
        assert np.all(error == 0)

    def test_generate_probability_half(self, code, error_model, rng):
        probability = 0.5
        error = error_model.generate(code, probability=probability, rng=rng)
        pauli = Toric3DPauli(code, bsf=error)
        indices = list(itertools.product(*[
            range(length) for length in code.size
        ]))
        for x, y, z in indices:
            assert pauli.operator((0, x, y, z)) == 'I', (
                'All x edges should have no error'
            )
            assert pauli.operator((1, x, y, z)) in ['I', 'X'], (
                'Any error on y edge must be only X error'
            )
            assert pauli.operator((2, x, y, z)) in ['I', 'X'], (
                'Any error on z edge must be only X error'
            )

        assert any(error), 'Error should be non-trivial'

        number_of_yz_edges = 2*len(indices)
        number_of_errors = bsf_wt(error)
        proportion_of_errors = number_of_errors/number_of_yz_edges
        assert abs(probability - proportion_of_errors) < 0.1, (
            'Number of errors on xy edges should reflect probability'
        )

    def test_generate_probability_one(self, code, error_model, rng):
        error = error_model.generate(code, probability=1, rng=rng)
        pauli = Toric3DPauli(code, bsf=error)
        indices = itertools.product(*[
            range(length) for length in code.size
        ])
        for x, y, z in indices:
            assert pauli.operator((0, x, y, z)) == 'I', (
                'All x edges should have no error'
            )
            assert pauli.operator((1, x, y, z)) == 'X', (
                'All y edges should have X'
            )
            assert pauli.operator((2, x, y, z)) == 'X', (
                'All z edges should have X'
            )
