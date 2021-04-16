"""
Test five qubit code very simple run.
"""

import pytest
import numpy as np
from bn3d.models import PauliErrorModel
import qecsim.paulitools as pt
from qecsim.models.basic import FiveQubitCode
from qecsim.models.generic import NaiveDecoder
from qecsim.paulitools import pauli_to_bsf


@pytest.fixture
def five_qubit_code():
    code = FiveQubitCode()
    return code


@pytest.fixture
def pauli_noise_model(five_qubit_code):
    error_model = PauliErrorModel((0.5, 0.1, 0.4))
    return error_model


def test_stabilizers(five_qubit_code):
    code = five_qubit_code
    expected_stabilizers = ['XZZXI', 'IXZZX', 'XIXZZ', 'ZXIXZ']
    assert np.all(code.stabilizers == np.array([
        pauli_to_bsf(pauli_string)
        for pauli_string in expected_stabilizers
    ]))


def test_logicals(five_qubit_code):
    code = five_qubit_code
    expected_logical_xs = ['XXXXX']
    expected_logical_zs = ['ZZZZZ']
    assert np.all(code.logical_xs == np.array([
        pauli_to_bsf(pauli_string)
        for pauli_string in expected_logical_xs
    ]))
    assert np.all(code.logical_zs == np.array([
        pauli_to_bsf(pauli_string)
        for pauli_string in expected_logical_zs
    ]))


def test_generate_noise(five_qubit_code, pauli_noise_model):
    code = five_qubit_code
    error_model = pauli_noise_model
    probability = 0.3
    np.random.seed(0)
    assert hasattr(error_model, 'generate')
    error = error_model.generate(code, probability, rng=np.random)
    assert error.shape == (10, )


def test_decoder(five_qubit_code, pauli_noise_model):
    code = five_qubit_code
    error_model = pauli_noise_model
    decoder = NaiveDecoder()
    assert hasattr(decoder, 'decode')

    # Want deterministic results.
    np.random.seed(0)

    # Generate a non-trivial error.
    probability = 0.3
    error = error_model.generate(code, probability, rng=np.random)
    assert np.any(error == 1)
    assert error.shape == (10, )

    # The syndrome should be non-trivial too.
    syndrome = pt.bsp(error, code.stabilizers.T)
    assert np.any(syndrome == 1)
    assert syndrome.shape == (code.stabilizers.shape[0], )

    correction = decoder.decode(code, syndrome)

    assert correction.shape == (10, )

    # The correction should have worked.
    total_error = (correction + error) % 2
    assert np.all(total_error == 0)
