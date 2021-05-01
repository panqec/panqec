import os
import pytest
import numpy as np
from qecsim.models.basic import FiveQubitCode
from qecsim.models.generic import NaiveDecoder
from bn3d.noise import PauliErrorModel
from bn3d.app import read_input_json, run_once, Simulation
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')


@pytest.fixture
def required_fields():
    return [
        'error', 'syndrome', 'correction', 'effective_error', 'success'
    ]


@pytest.mark.parametrize(
    'file_name, expected_runs',
    [
        ('single_input.json', 1),
        ('range_input.json', 27),
    ]
)
def test_read_json_input(file_name, expected_runs):
    single_input_json = os.path.join(DATA_DIR, file_name)
    batch_simulation = read_input_json(single_input_json)
    assert batch_simulation is not None
    assert len(batch_simulation._simulations) == expected_runs


def test_run_once(required_fields):
    code = FiveQubitCode()
    decoder = NaiveDecoder()
    direction = np.ones(3)/3
    error_model = PauliErrorModel(*direction)
    probability = 0.5
    results = run_once(
        code, error_model, decoder, error_probability=probability
    )
    assert set(results.keys()).issubset(required_fields)


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
        assert len(simulation._results) == 10
        assert all(
            set(results.keys()).issubset(required_fields)
            for results in simulation._results
        )
