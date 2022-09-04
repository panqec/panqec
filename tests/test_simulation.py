import os
import json
import shutil
from glob import glob
import pytest
import gzip
import numpy as np
from panqec.error_models import PauliErrorModel
from panqec.codes import Toric2DCode
from panqec.decoders import BeliefPropagationOSDDecoder
from panqec.simulation import (
    read_input_json, run_once, DirectSimulation, expand_input_ranges, run_file,
    merge_results_dicts, filter_legacy_params, BatchSimulation
)
from panqec.cli import merge_dirs
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
        ('single_input.json.gz', 1),
        ('range_input.json.gz', 27),
    ]
)
def test_read_json_input(file_name, expected_runs):
    input_json = os.path.join(DATA_DIR, file_name)
    batch_simulation = read_input_json(input_json)
    assert batch_simulation is not None
    assert len(batch_simulation._simulations) == expected_runs
    parameters = [
        {
            'code': list(s.code.size),
            'noise': s.error_model.direction,
            'probability': s.error_rate
        }
        for s in batch_simulation._simulations
    ]
    for i in range(len(parameters)):
        for j in range(len(parameters)):
            if i != j:
                assert parameters[i] != parameters[j]


class TestRunOnce:
    @pytest.fixture
    def code(self):
        return Toric2DCode(3, 3)

    @pytest.fixture
    def error_model(self):
        return PauliErrorModel(1/3, 1/3, 1/3)

    def test_run_once(self, required_fields, code, error_model):
        error_rate = 0.5
        decoder = BeliefPropagationOSDDecoder(code, error_model, error_rate)
        results = run_once(
            code, error_model, decoder, error_rate=error_rate
        )
        assert set(required_fields).issubset(results.keys())
        assert results['error'].shape == (2*code.n,)
        assert results['syndrome'].shape == (code.stabilizer_matrix.shape[0],)
        assert isinstance(results['success'], bool)
        assert results['effective_error'].shape == (
            2*code.logicals_x.shape[0],
        )
        assert isinstance(results['codespace'], bool)

    def test_run_once_invalid_probability(self, code, error_model):
        error_rate = -1
        decoder = BeliefPropagationOSDDecoder(
            code, error_model, error_rate=error_rate
        )
        with pytest.raises(ValueError):
            run_once(code, error_model, decoder, error_rate=error_rate)


class TestSimulationToric2DCode():

    error_rate = 0.5

    @pytest.fixture
    def code(self):
        return Toric2DCode(3, 3)

    @pytest.fixture
    def decoder(self, code, error_model):
        return BeliefPropagationOSDDecoder(code, error_model, self.error_rate)

    @pytest.fixture
    def error_model(self):
        direction = np.ones(3)/3
        return PauliErrorModel(*direction)

    def test_run(self, code, error_model, decoder, required_fields):
        error_rate = 0.5
        simulation = DirectSimulation(code, error_model, decoder, error_rate)
        simulation.run(10)
        assert len(simulation._results['success']) == 10
        assert set(required_fields).issubset(simulation._results.keys())


class TestBatchSimulationOneFile():

    n_trials: int = 5

    @pytest.fixture
    def output_dir(self, tmpdir):
        out_dir = os.path.join(tmpdir, 'output_dir')
        os.mkdir(out_dir)
        assert os.path.isdir(out_dir)

        assert len(os.listdir(out_dir)) == 0
        batch_sim = BatchSimulation(
            label='mylabel',
            save_frequency=1,
            output_dir=out_dir,
            onefile=True
        )

        for size, error_rate in [(3, 0.1), (4, 0.5)]:
            code = Toric2DCode(size, size)
            error_model = PauliErrorModel(1/3, 1/3, 1/3)
            decoder = BeliefPropagationOSDDecoder(
                code, error_model, error_rate
            )
            simulation = DirectSimulation(
                code, error_model, decoder, error_rate
            )
            batch_sim.append(simulation)

        assert len(batch_sim) == 2

        batch_sim.run(self.n_trials)
        batch_sim.save_results()

        return out_dir

    def test_output_to_one_file(self, output_dir):

        # Make sure this is the only file produced.
        combined_outfile = os.path.join(
            output_dir, 'mylabel', 'mylabel.json.gz'
        )
        assert os.path.isfile(combined_outfile)
        assert len(os.listdir(os.path.join(output_dir, 'mylabel'))) == 1
        with gzip.open(combined_outfile, 'rb') as gz:
            results = json.loads(gz.read().decode('utf-8'))

        # Check integrity of results.
        assert len(results) == 2
        expected_input_keys = [
            'size', 'code', 'n', 'k', 'd', 'error_model', 'decoder',
            'probability'
        ]
        expected_results_keys = [
            'effective_error', 'success', 'codespace', 'wall_time',
        ]
        for key in expected_input_keys:
            assert key in results[0]['inputs']
            assert key in results[1]['inputs']

        for key in expected_results_keys:
            assert key in results[0]['results']
            assert key in results[1]['results']

        # Numerical tests on the results.
        assert len(results[0]['results']['effective_error']) == self.n_trials
        assert len(results[0]['results']['success']) == self.n_trials
        assert len(results[0]['results']['codespace']) == self.n_trials

    @pytest.mark.skip
    def test_merge_output_dirs(self, tmpdir, output_dir):
        shutil.copytree(output_dir, os.path.join(tmpdir, 'results_1'))
        shutil.copytree(output_dir, os.path.join(tmpdir, 'results_2'))
        assert set(os.listdir(tmpdir)) == set([
            'output_dir', 'results_1', 'results_2'
        ])
        merge_dirs(
            os.path.join(tmpdir, 'results'),
            ' '.join([
                os.path.join(tmpdir, 'results_1'),
                os.path.join(tmpdir, 'results_2'),
            ])
        )
        assert os.path.exists(os.path.join(tmpdir, 'results'))


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


def test_run_file_range_input(tmpdir):
    input_json = os.path.join(DATA_DIR, 'range_input.json')
    n_trials = 2
    assert os.listdir(tmpdir) == []
    run_file(input_json, n_trials, output_dir=tmpdir)
    assert os.listdir(tmpdir) == ['test_range']
    assert len(os.listdir(os.path.join(tmpdir, 'test_range'))) > 0
    out_files = glob(os.path.join(tmpdir, 'test_range', '*.json.gz'))
    assert len(out_files) > 0


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
                'probability': 0.05
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
                'probability': 0.05
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
            'probability': 0.05
        }
    }

    merged_results = merge_results_dicts(results_dicts)

    assert merged_results == expected_merged_results


def test_filter_legacy_params():
    old_params = {
        'joschka': True
    }
    filtered_params = filter_legacy_params(old_params)
    assert 'joschka' not in filtered_params
