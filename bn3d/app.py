"""
API for running simulations.
"""
import os
import json
import itertools
from typing import List, Dict, Callable
import datetime
import numpy as np
from qecsim.model import StabilizerCode, ErrorModel, Decoder
from .bpauli import bcommute, get_effective_error
from .config import (
    codes, error_models, decoders, noise_dependent_decoders, BN3D_DIR
)
from .utils import identity, NumpyEncoder


def run_once(
    code: StabilizerCode,
    error_model: ErrorModel,
    decoder: Decoder,
    error_probability: float,
    rng=None
) -> dict:
    """Run a simulation once and return the results as a dictionary."""

    if not (0 <= error_probability <= 1):
        raise ValueError('Error probability must be in [0, 1].')

    if rng is None:
        rng = np.random.default_rng()

    error = error_model.generate(code, probability=error_probability)
    syndrome = bcommute(code.stabilizers, error)
    correction = decoder.decode(code, syndrome)
    total_error = (correction + error) % 2
    effective_error = get_effective_error(
        total_error, code.logical_xs, code.logical_zs
    )
    success = np.all(effective_error == 0)

    results = {
        'error': error,
        'syndrome': syndrome,
        'correction': correction,
        'effective_error': effective_error,
        'success': success,
    }

    return results


class Simulation:
    """Quantum Error Correction Simulation."""

    start_time: datetime.datetime
    code: StabilizerCode
    error_model: ErrorModel
    decoder: Decoder
    error_probability: float
    label: str
    _results: list = []
    rng = None

    def __init__(
        self, code: StabilizerCode, error_model: ErrorModel, decoder: Decoder,
        probability: float, rng=None
    ):
        self.code = code
        self.error_model = error_model
        self.decoder = decoder
        self.error_probability = probability
        self.rng = rng
        self.label = '_'.join([
            code.label, error_model.label, decoder.label, f'{probability}'
        ])

    def run(self, repeats: int):
        """Run assuming perfect measurement."""
        self.start_time = datetime.datetime.now()
        for i_trial in range(repeats):
            self._results.append(
                run_once(
                    self.code, self.error_model, self.decoder,
                    error_probability=self.error_probability,
                    rng=self.rng
                )
            )

    @property
    def file_name(self) -> str:
        file_name = self.label + '.json'
        return file_name

    def get_file_path(self, output_dir: str) -> str:
        file_path = os.path.join(output_dir, self.file_name)
        return file_path

    def load_results(self, output_dir: str):
        """Load results from directory."""
        file_path = self.get_file_path(output_dir)
        if os.path.exists(file_path):
            with open(file_path) as f:
                data = json.load(f)
            self._results = data['results']

    def save_results(self, output_dir: str):
        """Save results to directory."""
        with open(self.get_file_path(output_dir), 'w') as f:
            json.dump({
                'results': self._results,
                'inputs': {
                    'code': self.code.label,
                    'error_model': self.error_model.label,
                    'decoder': self.decoder.label,
                    'error_probability': self.error_probability,
                }
            }, f, cls=NumpyEncoder)


class BatchSimulation():

    _simulations: List[Simulation]
    update_frequency: int
    save_frequency: int
    _output_dir: str

    def __init__(
        self,
        label='unlabelled',
        on_update: Callable = identity,
        update_frequency: int = 10,
        save_frequency: int = 10,
    ):
        self._simulations = []
        self.update_frequency = update_frequency
        self.save_frequency = save_frequency
        self.label = label
        self._output_dir = os.path.join(BN3D_DIR, self.label)
        os.makedirs(self._output_dir, exist_ok=True)

    def append(self, simulation: Simulation):
        self._simulations.append(simulation)

    def load_results(self):
        for simulation in self._simulations:
            simulation.load_results(self._output_dir)

    def on_update(self):
        pass

    def run(self, n_trials, progress: Callable = identity):
        self.load_results()
        max_remaining_trials = max([
            max(0, n_trials - len(simulation._results))
            for simulation in self._simulations
        ])
        for i_trial in progress(list(range(max_remaining_trials))):
            for simulation in self._simulations:
                if len(simulation._results) < n_trials:
                    simulation.run(1)
            if i_trial > 0:
                if i_trial % self.update_frequency:
                    self.on_update()
                if i_trial % self.save_frequency:
                    self.save_results()
            if i_trial == n_trials - 1:
                self.on_update()
                self.save_results()

    def save_results(self):
        for simulation in self._simulations:
            simulation.save_results(self._output_dir)

    def get_results(self):
        results = []
        for simulation in self._simulations:
            results.append(simulation.results)
        return results


def _parse_parameters_range(parameters):
    parameters_range = [[]]
    if len(parameters) > 0:
        if isinstance(parameters, list):
            parameters_range = parameters
        else:
            parameters_range = [parameters]
    return parameters_range


def expand_inputs_ranges(data: dict) -> List[Dict]:
    runs: List[Dict] = []
    code_range: List[List] = [[]]
    if 'parameters' in data['code']:
        code_range = _parse_parameters_range(data['code']['parameters'])

    noise_range: List[List] = [[]]
    if 'parameters' in data['noise']:
        noise_range = _parse_parameters_range(data['noise']['parameters'])

    decoder_range: List[List] = [[]]
    if 'parameters' in data['decoder']:
        decoder_range = _parse_parameters_range(
            data['decoder']['parameters']
        )

    probability_range = _parse_parameters_range(data['probability'])

    for (
        code_param, noise_param, decoder_param, probability
    ) in itertools.product(
        code_range, noise_range, decoder_range, probability_range
    ):
        run = {
            k: v for k, v in data.items()
            if k not in ['code', 'noise', 'decoder', 'probability']
        }
        for key in ['code', 'noise', 'decoder']:
            run[key] = {
                k: v for k, v in data[key].items() if k != 'parameters'
            }
        run['code']['parameters'] = code_param
        run['noise']['parameters'] = noise_param
        run['decoder']['parameters'] = decoder_param
        run['probability'] = probability
        runs.append(run)

    return runs


def parse_run(run: dict) -> Simulation:
    """Parse a single dict describing the run."""
    code_name = run['code']['model']
    if 'parameters' in run['code']:
        code_params = run['code']['parameters']
    else:
        code_params = []
    code_class = codes[code_name]
    code = code_class(*code_params)

    error_model_name = run['noise']['model']
    if 'parameters' in run['noise']:
        error_model_params = run['noise']['parameters']
    else:
        error_model_params = []
    error_model_class = error_models[error_model_name]
    error_model = error_model_class(*error_model_params)

    probability = run['probability']

    decoder_name = run['decoder']['model']
    if 'parameters' in run['decoder']:
        decoder_params = run['decoder']['parameters']
    else:
        decoder_params = []
    decoder = _create_decoder(
        decoder_name, decoder_params, error_model, probability
    )
    simulation = Simulation(code, error_model, decoder, probability)
    return simulation


def _create_decoder(
    decoder_name: str,
    decoder_params: list = [],
    error_model: ErrorModel = None,
    probability: float = None
) -> Decoder:
    """Create decoder maybe depending on noise model and rate.

    Because some decoders can be optimized for a given noise model
    and error rate.
    """
    decoder_class = decoders[decoder_name]

    # TODO deal with decoders that depend on noise
    if decoder_name in noise_dependent_decoders:
        pass
    decoder = decoder_class(*decoder_params)
    return decoder


def read_input_json(file_path: str) -> BatchSimulation:
    """Read json input file."""
    with open(file_path) as f:
        data = json.load(f)
    return read_input_dict(data)


def read_input_dict(data: dict) -> BatchSimulation:
    """Return BatchSimulation from input dict."""
    runs = []
    label = 'unlabelled'
    if 'runs' in data:
        runs = data['runs']
    if 'ranges' in data:
        runs += expand_inputs_ranges(data['ranges'])
        if 'label' in data['ranges']:
            label = data['ranges']['label']

    batch_sim = BatchSimulation(label=label)
    assert len(batch_sim._simulations) == 0

    for single_run in runs:
        batch_sim.append(parse_run(single_run))

    return batch_sim
