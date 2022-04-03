"""
API for running simulations.
"""
import os
import inspect
import json
from json import JSONDecodeError
import itertools
from typing import List, Dict, Callable, Union, Any, Optional, Tuple
import datetime
import numpy as np
from panqec.codes import StabilizerCode
from panqec.decoders import BaseDecoder
from panqec.error_models import BaseErrorModel
from .bpauli import bcommute, get_effective_error
from .config import (
    CODES, ERROR_MODELS, DECODERS, PANQEC_DIR
)
from .utils import identity, NumpyEncoder


def run_once(
    code: StabilizerCode,
    error_model: BaseErrorModel,
    decoder: BaseDecoder,
    error_probability: float,
    rng=None
) -> dict:
    """Run a simulation once and return the results as a dictionary."""

    if not (0 <= error_probability <= 1):
        raise ValueError('Error probability must be in [0, 1].')

    if rng is None:
        rng = np.random.default_rng()

    error = error_model.generate(code, probability=error_probability, rng=rng)
    syndrome = bcommute(code.stabilizer_matrix, error)
    correction = decoder.decode(code, syndrome)
    total_error = correction + error
    total_error.data %= 2
    effective_error = get_effective_error(
        total_error, code.logicals_x, code.logicals_z
    )
    codespace = bool(np.all(bcommute(code.stabilizer_matrix, total_error) == 0))
    success = bool(np.all(effective_error == 0)) and codespace

    results = {
        'error': error,
        'syndrome': syndrome,
        'correction': correction,
        'effective_error': effective_error,
        'success': success,
        'codespace': codespace,
    }

    return results


def run_file(
    file_name: str, n_trials: int,
    start: Optional[int] = None,
    n_runs: Optional[int] = None,
    progress: Callable = identity,
    output_dir: Optional[str] = None,
    verbose: bool = True,
):
    """Run an input json file."""
    batch_sim = read_input_json(
        file_name, output_dir=output_dir,
        start=start, n_runs=n_runs
    )
    if verbose:
        print(f'running {len(batch_sim._simulations)} simulations:')
        for simulation in batch_sim._simulations:
            code = simulation.code.label
            noise = simulation.error_model.label
            decoder = simulation.decoder.label
            probability = simulation.error_probability
            print(f'    {code}, {noise}, {decoder}, {probability}')
    batch_sim.run(n_trials, progress=progress)


class Simulation:
    """Quantum Error Correction Simulation."""

    start_time: datetime.datetime
    code: StabilizerCode
    error_model: BaseErrorModel
    decoder: BaseDecoder
    error_probability: float
    label: str
    _results: dict = {}
    rng = None

    def __init__(
        self, code: StabilizerCode, error_model: BaseErrorModel, decoder: BaseDecoder,
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
        self._results = {
            'effective_error': [],
            'success': [],
            'codespace': [],
            'wall_time': 0,
        }

    @property
    def wall_time(self):
        return self._results['wall_time']

    def run(self, repeats: int):
        """Run assuming perfect measurement."""
        self.start_time = datetime.datetime.now()
        for i_trial in range(repeats):
            shot = run_once(
                self.code, self.error_model, self.decoder,
                error_probability=self.error_probability,
                rng=self.rng
            )
            for key, value in shot.items():
                if key in self._results.keys():
                    self._results[key].append(value)
        finish_time = datetime.datetime.now() - self.start_time
        self._results['wall_time'] += finish_time.total_seconds()

    @property
    def n_results(self):
        return len(self._results['success'])

    @property
    def results(self):
        res = self._results
        return res

    @property
    def file_name(self) -> str:
        file_name = self.label + '.json'
        return file_name

    def get_file_path(self, output_dir: str) -> str:
        file_path = os.path.join(output_dir, self.file_name)
        return file_path

    def load_results(self, output_dir: str):
        """Load previously written results from directory."""
        file_path = self.get_file_path(output_dir)
        try:
            if os.path.exists(file_path):
                with open(file_path) as f:
                    data = json.load(f)
                for key in self._results.keys():
                    if key in data['results'].keys():
                        self._results[key] = data['results'][key]
                        if (
                            isinstance(self.results[key], list)
                            and len(self.results[key]) > 0
                            and isinstance(self._results[key][0], list)
                        ):
                            self._results[key] = [
                                np.array(array_value)
                                for array_value in self._results[key]
                            ]
                self._results = data['results']
        except JSONDecodeError as err:
            print(f'Error loading existing results file {file_path}')
            print('Starting this from scratch')
            print(err)

    def save_results(self, output_dir: str):
        """Save results to directory."""
        with open(self.get_file_path(output_dir), 'w') as f:
            json.dump({
                'results': self._results,
                'inputs': {
                    'size': self.code.size,
                    'code': self.code.label,
                    'n': self.code.n,
                    'k': self.code.k,
                    'd': self.code.d,
                    'error_model': self.error_model.label,
                    'decoder': self.decoder.label,
                    'error_probability': self.error_probability,
                }
            }, f, cls=NumpyEncoder)

    def get_results(self):
        """Return results as dictionary."""

        success = np.array(self.results['success'])
        if len(success) > 0:
            n_fail = np.sum(~success)
        else:
            n_fail = 0
        simulation_data = {
            'size': self.code.size,
            'code': self.code.label,
            'n': self.code.n,
            'k': self.code.k,
            'd': self.code.d,
            'error_model': self.error_model.label,
            'probability': self.error_probability,
            'n_success': np.sum(success),
            'n_fail': n_fail,
            'n_trials': len(success),
        }

        # Use sample mean as estimator for effective error rate.
        if simulation_data['n_trials'] != 0:
            simulation_data['p_est'] = (
                simulation_data['n_fail']/simulation_data['n_trials']
            )
        else:
            simulation_data['p_est'] = np.nan

        # Use posterior Beta distribution of the effective error rate
        # standard distribution as standard error.
        simulation_data['p_se'] = np.sqrt(
            simulation_data['p_est']*(1 - simulation_data['p_est'])
            / (simulation_data['n_trials'] + 1)
        )
        return simulation_data


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
        save_frequency: int = 20,
        output_dir: Optional[str] = None,
    ):
        self._simulations = []
        self.update_frequency = update_frequency
        self.save_frequency = save_frequency
        self.label = label
        if output_dir is not None:
            self._output_dir = os.path.join(output_dir, self.label)
        else:
            self._output_dir = os.path.join(PANQEC_DIR, self.label)
        os.makedirs(self._output_dir, exist_ok=True)

    def __getitem__(self, *args):
        return self._simulations.__getitem__(*args)

    def __iter__(self):
        return self._simulations.__iter__()

    def __next__(self):
        return self._simulations.__next__()

    def append(self, simulation: Simulation):
        self._simulations.append(simulation)

    def load_results(self):
        for simulation in self._simulations:
            simulation.load_results(self._output_dir)

    def on_update(self):
        pass

    def estimate_remaining_time(self, n_trials: int):
        """Estimate remaining time given target n_trials."""
        return sum(
            (n_trials - sim.n_results)*sim.wall_time/sim.n_results
            for sim in self._simulations
            if sim.n_results != 0
        )

    @property
    def wall_time(self):
        return sum(sim.wall_time for sim in self._simulations)

    def run(self, n_trials, progress: Callable = identity):
        try:
            self._run(n_trials, progress=progress)
        except KeyboardInterrupt:
            print('Simulation paused')

    def _run(self, n_trials, progress: Callable = identity):
        self.load_results()

        # Use the maximum remaining trials to overestimate how much is
        # remaniing for purposes of reporting progress to give a conservative
        # estimate of what is left without disappointing the user.
        max_remaining_trials = max([
            max(0, n_trials - simulation.n_results)
            for simulation in self._simulations
        ])

        for i_trial in progress(list(range(max_remaining_trials))):
            for simulation in self._simulations:
                if simulation.n_results < n_trials:
                    simulation.run(1)
            if i_trial > 0:
                if i_trial % self.update_frequency == 0:
                    self.on_update()
                if i_trial % self.save_frequency == 0:
                    self.save_results()
            if i_trial == max_remaining_trials - 1:
                self.on_update()
                self.save_results()

    def _save_results(self):
        for simulation in self._simulations:
            simulation.save_results(self._output_dir)

    def save_results(self):
        try:
            self._save_results()

        # Do not give up saving results during keyboard interrupt.
        except KeyboardInterrupt:
            print('Simulation paused. Saving results. Do not interrupt again')
            self._save_results()
            print('Results saved')
            raise KeyboardInterrupt('Simulation paused')

    def get_results(self):
        results = []
        for simulation in self._simulations:
            simulation_data = simulation.get_results()
            results.append(simulation_data)
        return results


def _parse_parameters_range(parameters):
    parameters_range = [{}]
    if len(parameters) > 0:
        if isinstance(parameters, list):
            parameters_range = parameters
        elif isinstance(parameters, dict):
            parameters_range = [parameters]
        else:
            parameters_range = [parameters]
    return parameters_range


def _parse_all_ranges(data: dict) -> Tuple[list, list, list, list]:

    code_range: List[Dict] = [{}]
    if 'parameters' in data['code']:
        code_range = _parse_parameters_range(data['code']['parameters'])

    noise_range: List[Dict] = [{}]
    if 'parameters' in data['noise']:
        noise_range = _parse_parameters_range(data['noise']['parameters'])

    decoder_range: List[Dict] = [{}]
    if 'parameters' in data['decoder']:
        decoder_range = _parse_parameters_range(
            data['decoder']['parameters']
        )

    probability_range = _parse_parameters_range(data['probability'])
    return code_range, noise_range, decoder_range, probability_range


def expand_input_ranges(data: dict) -> List[Dict]:
    runs: List[Dict] = []

    (
        code_range, noise_range, decoder_range, probability_range
    ) = _parse_all_ranges(data)

    for (
        noise_param, decoder_param, probability, code_param
    ) in itertools.product(
        noise_range, decoder_range, probability_range, code_range
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


def _parse_code_dict(code_dict: Dict[str, Any]) -> StabilizerCode:
    code_name = code_dict['model']
    code_params: Union[list, dict] = []
    if 'parameters' in code_dict:
        code_params = code_dict['parameters']
    code_class = CODES[code_name]
    if isinstance(code_params, dict):
        code = code_class(**code_params)
    else:
        code = code_class(*code_params)
    return code


def _parse_error_model_dict(noise_dict: Dict[str, Any]) -> BaseErrorModel:
    error_model_name = noise_dict['model']
    error_model_params: Union[list, dict] = []
    if 'parameters' in noise_dict:
        error_model_params = noise_dict['parameters']
    error_model_class = ERROR_MODELS[error_model_name]
    if isinstance(error_model_params, dict):
        error_model = error_model_class(**error_model_params)
    else:
        error_model = error_model_class(*error_model_params)
    return error_model


def _parse_decoder_dict(
    decoder_dict: Dict[str, Any],
    error_model: BaseErrorModel,
    probability: float
) -> BaseDecoder:
    decoder_name = decoder_dict['model']
    decoder_class = DECODERS[decoder_name]
    decoder_params: dict = {}
    if 'parameters' in decoder_dict:
        decoder_params = decoder_dict['parameters']
    else:
        decoder_params = {}

    signature = inspect.signature(decoder_class)

    if 'error_model' in signature.parameters.keys():
        decoder_params['error_model'] = error_model
    if 'probability' in signature.parameters.keys():
        decoder_params['probability'] = probability
    decoder = decoder_class(**decoder_params)
    return decoder


def parse_run(run: Dict[str, Any]) -> Simulation:
    """Parse a single dict describing the run."""
    code = _parse_code_dict(run['code'])
    error_model = _parse_error_model_dict(run['noise'])
    probability = run['probability']
    decoder = _parse_decoder_dict(run['decoder'], error_model, probability)

    simulation = Simulation(code, error_model, decoder, probability)
    return simulation


def read_input_json(file_path: str, *args, **kwargs) -> BatchSimulation:
    """Read json input file."""
    try:
        with open(file_path) as f:
            data = json.load(f)
    except JSONDecodeError as err:
        print(f'Error reading input file {file_path}')
        raise err
    return read_input_dict(data, *args, **kwargs)


def get_runs(
    data: dict, start: Optional[int] = None, n_runs: Optional[int] = None
) -> List[dict]:
    """Get expanded runs from input dictionary."""
    runs = []
    if 'runs' in data:
        runs = data['runs']
    if 'ranges' in data:
        runs += expand_input_ranges(data['ranges'])

    # Filter the range of runs.
    if start is not None:
        runs = runs[start:]
        if n_runs is not None:
            runs = runs[:n_runs]

    return runs


def count_runs(file_path: str) -> Optional[int]:
    """Count the number of noise parameters in an input file.

    Return None if no noise parameters range given.
    """
    n_runs = None
    with open(file_path) as f:
        data = json.load(f)
    all_runs = get_runs(data, start=None, n_runs=None)
    n_runs = len(all_runs)
    return n_runs


def read_input_dict(
    data: dict,
    start: Optional[int] = None,
    n_runs: Optional[int] = None,
    *args, **kwargs
) -> BatchSimulation:
    """Return BatchSimulation from input dict."""
    label = 'unlabelled'
    if 'ranges' in data:
        if 'label' in data['ranges']:
            label = data['ranges']['label']
    kwargs['label'] = label

    batch_sim = BatchSimulation(*args, **kwargs)
    assert len(batch_sim._simulations) == 0

    runs = get_runs(data, start=start, n_runs=n_runs)

    for single_run in runs:
        batch_sim.append(parse_run(single_run))

    return batch_sim


def merge_results_dicts(results_dicts: List[Dict]) -> Dict:
    """Merge results dicts into one dict."""
    results = {
        'effective_error': [],
        'success': [],
        'codespace': [],
        'wall_time': 0.0
    }
    inputs = results_dicts[0]['inputs']
    for results_dict in results_dicts:
        for key in results.keys():
            results[key] += results_dict['results'][key]
        assert results_dict['inputs'] == inputs, (
            'Warning: attempting to merge results of different inputs'
        )

    merged_dict = {
        'results': results,
        'inputs': inputs,
    }
    return merged_dict
