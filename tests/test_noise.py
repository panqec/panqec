import numpy as np
import pytest
from panqec.bpauli import bsf_to_pauli, bsf_wt
from panqec.error_models import PauliErrorModel
from panqec.codes import Toric3DCode
from panqec.bsparse import to_array
from panqec.utils import get_direction_from_bias_ratio


class TestPauliNoise:

    @pytest.fixture(autouse=True)
    def seed_random(self):
        np.random.seed(0)

    @pytest.fixture
    def code(self):
        return Toric3DCode(3, 4, 5)

    @pytest.fixture
    def error_model(self):
        return PauliErrorModel(0.2, 0.3, 0.5)

    def test_label(self, error_model):
        assert error_model.label == 'Pauli X0.2000Y0.3000Z0.5000'

    def test_generate(self, code, error_model):
        error_rate = 0.1
        error = to_array(
            error_model.generate(code, error_rate, rng=np.random)
        )
        assert np.any(error != 0), 'Error should be non-trivial'
        assert error.shape == (2*code.n, ), 'Shape incorrect'

    def test_probability_zero(self, code, error_model):
        error_rate = 0
        error = to_array(
            error_model.generate(code, error_rate, rng=np.random)
        )
        assert np.all(error == 0), 'Should have no error'

    def test_probability_one(self, code, error_model):
        error_rate = 1
        error = error_model.generate(code, error_rate, rng=np.random)

        # Error everywhere so weight is number of qubits.
        assert bsf_wt(error) == code.n, 'Should be error everywhere'

    def test_generate_all_X_errors(self, code):
        error_rate = 1
        direction = (1, 0, 0)
        error_model = PauliErrorModel(*direction)
        error = error_model.generate(code, error_rate, rng=np.random)
        assert bsf_to_pauli(error) == 'X'*code.n, (
            'Should be X error everywhere'
        )

    def test_generate_all_Y_errors(self, code):
        error_rate = 1
        direction = (0, 1, 0)
        error_model = PauliErrorModel(*direction)
        error = error_model.generate(code, error_rate, rng=np.random)
        assert bsf_to_pauli(error) == 'Y'*code.n, (
            'Should be Y error everywhere'
        )

    def test_generate_all_Z_errors(self, code):
        error_rate = 1
        direction = (0, 0, 1)
        error_model = PauliErrorModel(*direction)
        error = error_model.generate(code, error_rate, rng=np.random)
        assert bsf_to_pauli(error) == 'Z'*code.n, (
            'Should be Z error everywhere'
        )

    def test_raise_error_if_direction_does_not_sum_to_1(self):
        with pytest.raises(ValueError):
            PauliErrorModel(0, 0, 0)


class TestGeneratePauliNoise:

    def generate_pauli_noise(self, p_X, p_Y, p_Z, L):
        code = Toric3DCode(L, L, L)
        error_rate = p_X + p_Y + p_Z
        if error_rate != 0:
            r_x, r_y, r_z = p_X/error_rate, p_Y/error_rate, p_Z/error_rate
        else:
            r_x, r_y, r_z = 1/3, 1/3, 1/3
        error_model = PauliErrorModel(r_x, r_y, r_z)
        noise = error_model.generate(code, error_rate)
        return noise

    @pytest.fixture(autouse=True)
    def seed_random(self):
        np.random.seed(0)

    def test_generate_pauli_noise(self):
        p_X = 0.1
        p_Y = 0.2
        p_Z = 0.5
        L = 10
        noise = self.generate_pauli_noise(p_X, p_Y, p_Z, L)
        assert list(noise.shape) == [2*3*L**3]
        assert np.issubdtype(noise.dtype, np.unsignedinteger)

    def test_no_errors_if_p_zero(self):
        L = 10
        noise = self.generate_pauli_noise(0, 0, 0, L)
        assert np.all(noise == 0)

    def test_all_X_if_p_X_one(self):
        L = 10
        noise = self.generate_pauli_noise(1, 0, 0, L)
        assert np.all(noise[:3*L**3] == 1)
        assert np.all(noise[3*L**3:] == 0)

    def test_all_Y_if_p_Y_one(self):
        L = 10
        noise = self.generate_pauli_noise(0, 1, 0, L)
        assert np.all(noise[:3*L**3] == 1)
        assert np.all(noise[3*L**3:] == 1)

    def test_all_Z_if_p_Z_one(self):
        L = 10
        noise = self.generate_pauli_noise(0, 0, 1, L)
        assert np.all(noise[:3*L**3] == 0)
        assert np.all(noise[3*L**3:] == 1)

    def test_only_X_if_p_X_only(self):
        L = 10
        noise = self.generate_pauli_noise(0.5, 0, 0, L)
        assert np.any(noise[:3*L**3] == 1)
        assert np.all(noise[3*L**3:] == 0)

    def test_only_Y_if_p_Y_only(self):
        L = 10
        noise = self.generate_pauli_noise(0, 0.5, 0, L)
        assert np.all(noise[:3*L**3] == noise[3*L**3:])

    def test_only_Z_if_p_Z_only(self):
        L = 10
        noise = self.generate_pauli_noise(0, 0, 0.5, L)
        assert np.all(noise[:3*L**3] == 0)
        assert np.any(noise[3*L**3:] == 1)


class TestDeformedMatchingWeights:

    def get_weights(self, p_X, p_Y, p_Z, L):
        code = Toric3DCode(L, L, L)
        error_rate = p_X + p_Y + p_Z
        if error_rate != 0:
            r_x, r_y, r_z = p_X/error_rate, p_Y/error_rate, p_Z/error_rate
        else:
            r_x, r_y, r_z = 1/3, 1/3, 1/3
        error_model = PauliErrorModel(
            r_x, r_y, r_z,
            deformation_name='XZZX',
            deformation_kwargs={'deformation_axis': 'z'}
        )
        weights_x, weight_z = error_model.get_weights(code, error_rate)
        return weights_x, weight_z

    def test_if_equal_rates_then_equal_weights(self):
        L = 10
        p_X, p_Y, p_Z = 0.1, 0.1, 0.1
        wx, wz = self.get_weights(p_X, p_Y, p_Z, L)
        assert np.all(wx != 0)
        assert np.all(wx == wx[0])

        assert np.all(wz != 0)
        assert np.all(wz == wz[0])

    def test_zero_error_rate_no_nan(self):
        L = 10
        p_X, p_Y, p_Z = 0, 0, 0
        wx, wz = self.get_weights(p_X, p_Y, p_Z, L)
        assert np.all(wx != 0)
        assert np.all(~np.isnan(wx))

        assert np.all(wz != 0)
        assert np.all(~np.isnan(wz))

    def test_one_error_rate_no_nan(self):
        L = 10
        p_X, p_Y, p_Z = 0.45, 0, 0
        wx, wz = self.get_weights(p_X, p_Y, p_Z, L)
        assert np.all(wx != 0)
        assert np.all(~np.isnan(wx))

    def test_biased_Z_noise_different_weights(self):
        L = 10
        p_X, p_Y, p_Z = 0, 0, 0.4
        wx, wz = self.get_weights(p_X, p_Y, p_Z, L)
        assert np.any(wx != 0)
        assert np.any(wx != wx[0])

    def test_only_x_edges_different_wx(self):
        L = 10
        p_X, p_Y, p_Z = 0.3, 0.1, 0
        wx, wz = self.get_weights(p_X, p_Y, p_Z, L)
        assert np.any(wx != wx[0])
        code = Toric3DCode(L, L, L)
        x_edge_indices = [
            index for index, edge in enumerate(code.qubit_index)
            if code.qubit_axis(edge) == 'x'
        ]
        y_edge_indices = [
            index for index, edge in enumerate(code.qubit_index)
            if code.qubit_axis(edge) == 'y'
        ]
        z_edge_indices = [
            index for index, edge in enumerate(code.qubit_index)
            if code.qubit_axis(edge) == 'z'
        ]

        # Weights for the same wedge type should be equal.
        assert np.all(wx[x_edge_indices] == wx[x_edge_indices[0]])
        assert np.all(wx[y_edge_indices] == wx[y_edge_indices[0]])
        assert np.all(wx[z_edge_indices] == wx[z_edge_indices[0]])

        # Weights for y-edges and z-edges should be equal.
        assert np.all(wx[x_edge_indices] == wx[y_edge_indices])

        # Weights for x-edges and z-edges should not be equal.
        assert np.any(wx[x_edge_indices] != wx[z_edge_indices])


@pytest.mark.parametrize('pauli,bias,expected', [
    ('X', 0.5, (1/3, 1/3, 1/3)),
    ('Y', 0.5, (1/3, 1/3, 1/3)),
    ('Z', 0.5, (1/3, 1/3, 1/3)),
    ('X', np.inf, (1, 0, 0)),
    ('Y', np.inf, (0, 1, 0)),
    ('Z', np.inf, (0, 0, 1)),
    ('X', 1, (0.5, 0.25, 0.25)),
    ('Y', 1, (0.25, 0.5, 0.25)),
    ('Z', 1, (0.25, 0.25, 0.5)),
])
def test_get_direction_from_bias_ratio(pauli, bias, expected):
    expected_params = dict(zip(['r_x', 'r_y', 'r_z'], expected))
    params = get_direction_from_bias_ratio(pauli, bias)
    for key in expected_params.keys():
        assert np.isclose(params[key], expected_params[key])
