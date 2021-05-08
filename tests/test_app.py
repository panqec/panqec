import os
import json
import pytest
import numpy as np
from qecsim.models.basic import FiveQubitCode
from qecsim.models.generic import NaiveDecoder
from bn3d.noise import PauliErrorModel
from bn3d.app import (
    read_input_json, run_once, Simulation, expand_inputs_ranges
)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')


@pytest.fixture
def required_fields():
    return [
        'effective_error', 'success'
    ]


@pytest.mark.parametrize(
    'file_name, expected_runs',
    [
        ('single_input.json', 1),
        ('range_input.json', 27),
    ]
)
def test_read_json_input(file_name, expected_runs):
    input_json = os.path.join(DATA_DIR, file_name)
    batch_simulation = read_input_json(input_json)
    assert batch_simulation is not None
    assert len(batch_simulation._simulations) == expected_runs
    parameters = [
        {
            'code': s.code.size,
            'noise': s.error_model.direction,
            'probability': s.error_probability
        }
        for s in batch_simulation._simulations
    ]
    for i in range(len(parameters)):
        for j in range(len(parameters)):
            if i != j:
                assert parameters[i] != parameters[j]


def test_run_once(required_fields):
    code = FiveQubitCode()
    decoder = NaiveDecoder()
    direction = np.ones(3)/3
    error_model = PauliErrorModel(*direction)
    probability = 0.5
    results = run_once(
        code, error_model, decoder, error_probability=probability
    )
    assert set(required_fields).issubset(results.keys())
    assert results['error'].shape == (2*code.n_k_d[0],)
    assert results['syndrome'].shape == (code.stabilizers.shape[0],)
    assert isinstance(results['success'], bool)
    assert results['effective_error'].shape == (2*code.logical_xs.shape[0],)


def test_run_once_invalid_probability():
    code = FiveQubitCode()
    decoder = NaiveDecoder()
    error_model = PauliErrorModel(1, 0, 0)
    probability = -1
    with pytest.raises(ValueError):
        run_once(code, error_model, decoder, error_probability=probability)


class TestSimulationFiveQubitCode():

    @pytest.fixture
    def code(self):
        return FiveQubitCode()

    @pytest.fixture
    def decoder(self):
        return NaiveDecoder()

    @pytest.fixture
    def error_model(self):
        direction = np.ones(3)/3
        return PauliErrorModel(*direction)

    def test_run(self, code, error_model, decoder, required_fields):
        probability = 0.5
        simulation = Simulation(code, error_model, decoder, probability)
        simulation.run(10)
        assert len(simulation._results['success']) == 10
        assert set(required_fields).issubset(simulation._results.keys())


@pytest.fixture
def example_ranges():
    input_json = os.path.join(DATA_DIR, 'range_input.json')
    with open(input_json) as f:
        data = json.load(f)
    return data['ranges']


def test_expand_inputs_ranges(example_ranges):
    ranges = example_ranges
    expanded_inputs = expand_inputs_ranges(ranges)
    assert len(expanded_inputs) == 27
