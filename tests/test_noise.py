import numpy as np
import pytest
from qecsim.paulitools import bsf_to_pauli, bsf_wt
from panqec.error_models import PauliErrorModel
from panqec.codes import Toric3DCode
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
