"""
Test five qubit code using qecsim classes.

:Author:
    Eric Huang
"""

import json
import pytest
import numpy as np
import qecsim.paulitools as pt
from qecsim.models.basic import FiveQubitCode
from qecsim.models.generic import NaiveDecoder
from qecsim.paulitools import pauli_to_bsf, bsf_to_pauli
from qecsim.app import run
from panqec.error_models import PauliErrorModel


@pytest.fixture
def five_qubit_code():
    code = FiveQubitCode()
    return code


@pytest.fixture
def pauli_noise_model():
    error_model = PauliErrorModel(0.5, 0.1, 0.4)
    return error_model


@pytest.fixture
def naive_decoder():
    return NaiveDecoder()


def test_stabilizers(five_qubit_code):
    code = five_qubit_code
    expected_stabilizers = ['XZZXI', 'IXZZX', 'XIXZZ', 'ZXIXZ']
    assert np.all(code.stabilizer_matrix == np.array([
        pauli_to_bsf(pauli_string)
        for pauli_string in expected_stabilizers
    ]))


def test_logicals(five_qubit_code):
    code = five_qubit_code
    expected_logicals_x = ['XXXXX']
    expected_logicals_z = ['ZZZZZ']
    assert np.all(code.logicals_x == np.array([
        pauli_to_bsf(pauli_string)
        for pauli_string in expected_logicals_x
    ]))
    assert np.all(code.logicals_z == np.array([
        pauli_to_bsf(pauli_string)
        for pauli_string in expected_logicals_z
    ]))


def test_generate_noise(five_qubit_code, pauli_noise_model):
    code = five_qubit_code
    error_model = pauli_noise_model
    probability = 0.3
    np.random.seed(0)
    assert hasattr(error_model, 'generate')
    error = error_model.generate(code, probability, rng=np.random)
    assert error.shape == (10, )


def test_decoder(five_qubit_code, pauli_noise_model, naive_decoder):
    code = five_qubit_code
    error_model = pauli_noise_model
    decoder = naive_decoder
    assert hasattr(decoder, 'decode')

    # Want deterministic results.
    np.random.seed(0)

    # Generate a non-trivial error.
    probability = 0.3
    error = error_model.generate(code, probability, rng=np.random)
    assert np.all(error == [
        0, 1, 0, 0, 0,
        0, 0, 0, 0, 0,
    ])
    assert bsf_to_pauli(error) == 'IXIII'
    assert np.any(error == 1)
    assert error.shape == (10, )

    # The syndrome should be non-trivial too.
    syndrome = pt.bsp(error, code.stabilizer_matrix.T)
    assert np.any(syndrome == 1)
    assert syndrome.shape == (code.stabilizer_matrix.shape[0], )

    correction = decoder.decode(code, syndrome)
    assert bsf_to_pauli(correction) == 'IXIII'

    assert correction.shape == (10, )

    # The correction should have worked.
    total_error = (correction + error) % 2
    assert np.all(total_error == 0)
    assert bsf_to_pauli(total_error) == 'IIIII'


def test_run(five_qubit_code, pauli_noise_model, naive_decoder):
    code = five_qubit_code
    error_model = pauli_noise_model
    decoder = naive_decoder
    error_probability = 0.4
    results = run(
        code, error_model, decoder, error_probability, max_runs=None,
        max_failures=None, random_seed=0
    )
    assert type(results) is dict
    assert sorted(results.keys()) == [
        'code',
        'custom_totals',
        'decoder',
        'error_model',
        'error_probability',
        'error_weight_pvar',
        'error_weight_total',
        'logical_failure_rate',
        'measurement_error_probability',
        'n_fail',
        'n',
        'k',
        'd',
        'n_logical_commutations',
        'n_run',
        'n_success',
        'physical_error_rate',
        'time_steps',
        'wall_time',
    ]
    assert results['n'] == 5
    assert results['k'] == 1
    assert results['d'] == 3
    assert len(results['n_logical_commutations']) == 2
    assert results['code'] == '5-qubit'

    json_string = json.dumps(results)
    parsed_results = json.loads(json_string)
    assert results.keys() == parsed_results.keys()
