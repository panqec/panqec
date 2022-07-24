"""
API for running simulations.
"""
import os
import json
from json import JSONDecodeError
import itertools
from typing import List, Dict, Callable, Union, Any, Optional, Tuple
import datetime
import numpy as np
from panqec.codes import StabilizerCode
from panqec.decoders import BaseDecoder
from panqec.error_models import BaseErrorModel
from .config import (
    CODES, ERROR_MODELS, DECODERS, PANQEC_DIR
)
from .utils import identity, NumpyEncoder


def run_once(
    code: StabilizerCode,
    error_model: BaseErrorModel,
    decoder: BaseDecoder,
    error_rate: float,
    rng=None
) -> dict:
    """Run a simulation once and return the results as a dictionary."""

    if not (0 <= error_rate <= 1):
        raise ValueError('Error rate must be in [0, 1].')

    if rng is None:
        rng = np.random.default_rng()

    error = error_model.generate(code, error_rate=error_rate, rng=rng)
    syndrome = code.measure_syndrome(error)
    correction = decoder.decode(syndrome)
    total_error = (correction + error) % 2
    effective_error = code.logical_errors(total_error)
    codespace = code.in_codespace(total_error)
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
            error_rate = simulation.error_rate
            print(f'    {code}, {noise}, {decoder}, {error_rate}')
    batch_sim.run(n_trials, progress=progress)


class Simulation:
    """Quantum Error Correction Simulation."""

    start_time: datetime.datetime
    code: StabilizerCode
    error_model: BaseErrorModel
    decoder: BaseDecoder
    error_rate: float
    label: str
    _results: dict = {}
    rng = None

    def __init__(
        self,
        code: StabilizerCode,
        error_model: BaseErrorModel,
        decoder: BaseDecoder,
        error_rate: float, rng=None
    ):
        self.code = code
        self.error_model = error_model
        self.decoder = decoder
        self.error_rate = error_rate
        self.rng = rng
        self.label = '_'.join([
            code.label, error_model.label, decoder.label, f'{error_rate}'
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
                error_rate=self.error_rate,
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
                    'probability': self.error_rate,
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
            'probability': self.error_rate,
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
        self.code: Dict = {}
        self.decoder: Dict = {}
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
        # remaining for purposes of reporting progress to give a conservative
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
    if 'parameters' in data['code']:
        params_range = _parse_parameters_range(data['code']['parameters'])
        code_range = []
        for params in params_range:
            code_range.append(data['code'].copy())
            code_range[-1]['parameters'] = params

    noise_range: List[Dict] = [{}]
    if 'parameters' in data['noise']:
        params_range = _parse_parameters_range(data['noise']['parameters'])
        noise_range = []
        for params in params_range:
            noise_range.append(data['noise'].copy())
            noise_range[-1]['parameters'] = params

    decoder_range: List[Dict] = [{}]
    parameters = []
    if 'parameters' in data['decoder']:
        parameters = data['decoder']['parameters']

    params_range = _parse_parameters_range(parameters)
    decoder_range = []
    for params in params_range:
        decoder_range.append(data['decoder'].copy())
        decoder_range[-1]['parameters'] = params

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
        run['code']['parameters'] = code_param['parameters']
        run['noise']['parameters'] = noise_param['parameters']
        run['decoder']['parameters'] = decoder_param['parameters']
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
    code: StabilizerCode,
    error_model: BaseErrorModel,
    error_rate: float
) -> BaseDecoder:
    decoder_name = decoder_dict['model']
    decoder_class = DECODERS[decoder_name]
    decoder_params: dict = {}
    if 'parameters' in decoder_dict:
        decoder_params = decoder_dict['parameters']
    else:
        decoder_params = {}

    decoder_params['code'] = code
    decoder_params['error_model'] = error_model
    decoder_params['error_rate'] = error_rate

    filtered_decoder_params = filter_legacy_params(decoder_params)
    decoder = decoder_class(**filtered_decoder_params)
    return decoder


def parse_run(run: Dict[str, Any]) -> Simulation:
    """Parse a single dict describing the run."""
    code = _parse_code_dict(run['code'])
    error_model = _parse_error_model_dict(run['noise'])
    error_rate = run['probability']
    decoder = _parse_decoder_dict(
        run['decoder'], code, error_model, error_rate
    )

    simulation = Simulation(code, error_model, decoder, error_rate)
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


def get_simulations(
    data: dict, start: Optional[int] = None, n_runs: Optional[int] = None
) -> List[dict]:
    simulations = []

    if 'ranges' in data:
        (
            code_range, noise_range, decoder_range, probability_range
        ) = _parse_all_ranges(data['ranges'])

        codes = [_parse_code_dict(code_dict) for code_dict in code_range]
        error_models = [_parse_error_model_dict(noise_dict)
                        for noise_dict in noise_range]

    elif 'runs' in data:
        codes = [_parse_code_dict(run['code']) for run in data['runs']]
        decoder_range = [run['decoder'] for run in data['runs']]
        error_models = [_parse_error_model_dict(run['noise'])
                        for run in data['runs']]
        probability_range = [run['probability'] for run in data['runs']]

    else:
        raise ValueError("Invalid data format: does not have 'runs'\
                         or 'ranges' key")

    for (
        code, error_model, decoder_dict, error_rate
    ) in itertools.product(codes,
                           error_models,
                           decoder_range,
                           probability_range):
        decoder = _parse_decoder_dict(decoder_dict, code, error_model,
                                      error_rate)
        simulations.append(Simulation(code, error_model, decoder, error_rate))

    if start is not None:
        simulations = simulations[start:]
    if n_runs is not None:
        simulations = simulations[:n_runs]

    return simulations


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

    simulations = get_simulations(data, start=start, n_runs=n_runs)

    for sim in simulations:
        batch_sim.append(sim)

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


def filter_legacy_params(decoder_params: Dict[str, Any]) -> Dict[str, Any]:
    """Filter legacy parameters for decoders from old data."""
    new_params = decoder_params.copy()
    if 'joschka' in decoder_params:
        new_params.pop('joschka')
    return new_params
