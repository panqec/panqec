import os
import json
import pytest
import numpy as np
from panqec.error_models import PauliErrorModel
from panqec.app import (
    read_input_json, run_once, Simulation, expand_input_ranges, run_file,
    merge_results_dicts
)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')


@pytest.fixture
def required_fields():
    return [
        'effective_error', 'success'
    ]


@pytest.mark.skip(reason='sparse')
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
            'code': s.code.size.tolist(),
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
    assert results['error'].shape == (2*code.n,)
    assert results['syndrome'].shape == (code.stabilizer_matrix.shape[0],)
    assert isinstance(results['success'], bool)
    assert results['effective_error'].shape == (2*code.logicals_x.shape[0],)
    assert isinstance(results['codespace'], bool)


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
    expanded_inputs = expand_input_ranges(ranges)
    assert len(expanded_inputs) == 27


@pytest.mark.skip(reason='sparse')
def test_run_file_range_input(tmpdir):
    input_json = os.path.join(DATA_DIR, 'range_input.json')
    n_trials = 2
    assert os.listdir(tmpdir) == []
    run_file(input_json, n_trials, output_dir=tmpdir)
    assert os.listdir(tmpdir) == ['test_range']
    assert len(os.listdir(os.path.join(tmpdir, 'test_range'))) > 0


def test_merge_results():
    results_dicts = [
        {
            'results': {
                'effective_error': [[0, 0]],
                'success': [True],
                'codespace': [True],
                'wall_time': 0.018024
            },
            'inputs': {
                'size': [2, 2, 2],
                'code': 'Rotated Planar 2x2x2',
                'n': 99,
                'k': 1,
                'd': -1,
                'error_model': 'Pauli X0.2500Y0.2500Z0.5000',
                'decoder': 'BP-OSD decoder',
                'error_probability': 0.05
            }
        },
        {
            'results': {
                'effective_error': [[0, 1]],
                'success': [False],
                'codespace': [False],
                'wall_time': 0.017084
            },
            'inputs': {
                'size': [2, 2, 2],
                'code': 'Rotated Planar 2x2x2',
                'n': 99,
                'k': 1,
                'd': -1,
                'error_model': 'Pauli X0.2500Y0.2500Z0.5000',
                'decoder': 'BP-OSD decoder',
                'error_probability': 0.05
            }
        }
    ]
    expected_merged_results = {
        'results': {
            'effective_error': [[0, 0], [0, 1]],
            'success': [True, False],
            'codespace': [True, False],
            'wall_time': 0.035108
        },
        'inputs': {
            'size': [2, 2, 2],
            'code': 'Rotated Planar 2x2x2',
            'n': 99,
            'k': 1,
            'd': -1,
            'error_model': 'Pauli X0.2500Y0.2500Z0.5000',
            'decoder': 'BP-OSD decoder',
            'error_probability': 0.05
        }
    }

    merged_results = merge_results_dicts(results_dicts)
    assert merged_results == expected_merged_results
