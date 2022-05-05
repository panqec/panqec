import numpy as np
import pytest
from panqec.bpauli import bsf_to_pauli, bsf_wt
from panqec.error_models import PauliErrorModel, DeformedXZZXErrorModel
from panqec.codes import Toric3DCode
from panqec.decoders import DeformedToric3DPymatchingDecoder
import panqec.bsparse as bsparse
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
        probability = 0.1
        error = bsparse.to_array(error_model.generate(code, probability, rng=np.random))
        assert np.any(error != 0), 'Error should be non-trivial'
        assert error.shape == (2*code.n, ), 'Shape incorrect'

    def test_probability_zero(self, code, error_model):
        probability = 0
        error = bsparse.to_array(error_model.generate(code, probability, rng=np.random))
        assert np.all(error == 0), 'Should have no error'

    def test_probability_one(self, code, error_model):
        probability = 1
        error = error_model.generate(code, probability, rng=np.random)

        # Error everywhere so weight is number of qubits.
        assert bsf_wt(error) == code.n, 'Should be error everywhere'

    def test_generate_all_X_errors(self, code):
        probability = 1
        direction = (1, 0, 0)
        error_model = PauliErrorModel(*direction)
        error = error_model.generate(code, probability, rng=np.random)
        assert bsf_to_pauli(error) == 'X'*code.n, (
            'Should be X error everywhere'
        )

    def test_generate_all_Y_errors(self, code):
        probability = 1
        direction = (0, 1, 0)
        error_model = PauliErrorModel(*direction)
        error = error_model.generate(code, probability, rng=np.random)
        assert bsf_to_pauli(error) == 'Y'*code.n, (
            'Should be Y error everywhere'
        )

    def test_generate_all_Z_errors(self, code):
        probability = 1
        direction = (0, 0, 1)
        error_model = PauliErrorModel(*direction)
        error = error_model.generate(code, probability, rng=np.random)
        assert bsf_to_pauli(error) == 'Z'*code.n, (
            'Should be Z error everywhere'
        )

    def test_raise_error_if_direction_does_not_sum_to_1(self):
        with pytest.raises(ValueError):
            PauliErrorModel(0, 0, 0)


class TestGeneratePauliNoise:

    def generate_pauli_noise(self, p_X, p_Y, p_Z, L):
        code = Toric3DCode(L, L, L)
        probability = p_X + p_Y + p_Z
        if probability != 0:
            r_x, r_y, r_z = p_X/probability, p_Y/probability, p_Z/probability
        else:
            r_x, r_y, r_z = 1/3, 1/3, 1/3
        error_model = PauliErrorModel(r_x, r_y, r_z)
        noise = error_model.generate(code, probability)
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


@pytest.mark.skip(reason='superseded')
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


class TestDeformedMatchingWeights:

    def get_deformed_weights(self, p_X, p_Y, p_Z, L):
        code = Toric3DCode(L, L, L)
        probability = p_X + p_Y + p_Z
        if probability != 0:
            r_x, r_y, r_z = p_X/probability, p_Y/probability, p_Z/probability
        else:
            r_x, r_y, r_z = 1/3, 1/3, 1/3
        error_model = DeformedXZZXErrorModel(r_x, r_y, r_z)
        decoder = DeformedToric3DPymatchingDecoder(error_model, probability)
        weights = decoder.get_deformed_weights(code)
        return weights

    def test_if_equal_rates_then_equal_weights(self):
        L = 10
        p_X, p_Y, p_Z = 0.1, 0.1, 0.1
        weights = self.get_deformed_weights(p_X, p_Y, p_Z, L)
        assert np.all(weights != 0)
        assert np.all(weights == weights[0])

    def test_zero_error_rate_no_nan(self):
        L = 10
        p_X, p_Y, p_Z = 0, 0, 0
        weights = self.get_deformed_weights(p_X, p_Y, p_Z, L)
        assert np.all(weights != 0)
        assert np.all(~np.isnan(weights))

    def test_one_error_rate_no_nan(self):
        L = 10
        p_X, p_Y, p_Z = 1, 0, 0
        weights = self.get_deformed_weights(p_X, p_Y, p_Z, L)
        assert np.all(weights != 0)
        assert np.all(~np.isnan(weights))

    def test_biased_Z_noise_different_weights(self):
        L = 10
        p_X, p_Y, p_Z = 0, 0, 0.4
        weights = self.get_deformed_weights(p_X, p_Y, p_Z, L)
        assert np.any(weights != 0)
        assert np.any(weights != weights[0])

    @pytest.mark.skip(reason='refactor')
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


@pytest.mark.skip(reason='superseded')
class TestXNoiseOnYZEdgesOnly:

    @pytest.fixture(autouse=True)
    def rng(self):
        return np.random.default_rng(seed=0)

    @pytest.fixture
    def code(self):
        return Toric3DCode(3, 4, 5)

    @pytest.fixture
    def error_model(self):
        return XNoiseOnYZEdgesOnly()

    def test_label(self, error_model):
        assert error_model.label == 'X on yz edges'

    def test_generate_zero_probability(self, code, error_model, rng):
        error = error_model.generate(code, probability=0, rng=rng)

        # Sparse array number of non-zero elements is zero.
        assert error.nnz == 0

    def test_generate_probability_half(self, code, error_model, rng):
        probability = 0.5
        error = error_model.generate(code, probability=probability, rng=rng)
        pauli = Toric3DPauli(code, bsf=error)

        number_of_yz_edges = 0

        for edge in code.qubit_index:
            axis = code.axis(edge)
            if axis == code.X_AXIS:
                assert pauli.operator(edge) == 'I', (
                    'All x edges should have no error'
                )
            elif axis == code.Y_AXIS:
                number_of_yz_edges += 1
                assert pauli.operator(edge) in ['I', 'X'], (
                    'Any error on y edge must be only X error'
                )
            elif axis == code.Z_AXIS:
                number_of_yz_edges += 1
                assert pauli.operator(edge) in ['I', 'X'], (
                    'Any error on z edge must be only X error'
                )

        assert error.nnz != 0, 'Error should be non-trivial'

        number_of_errors = bsf_wt(error)
        proportion_of_errors = number_of_errors/number_of_yz_edges
        assert abs(probability - proportion_of_errors) < 0.1, (
            'Number of errors on xy edges should reflect probability'
        )

    def test_generate_probability_one(self, code, error_model, rng):
        error = error_model.generate(code, probability=1, rng=rng)
        pauli = Toric3DPauli(code, bsf=error)
        for edge in code.qubit_index:
            direction = tuple(np.mod(edge, 2).tolist())
            if direction == (1, 0, 0):
                assert pauli.operator(edge) == 'I', (
                    'All x edges should have no error'
                )
            elif direction == (0, 1, 0):
                assert pauli.operator(edge) == 'X', (
                    'All y edges should have X'
                )
            elif direction == (0, 0, 1):
                assert pauli.operator(edge) == 'X', (
                    'All z edges should have X'
                )


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
