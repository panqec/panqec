"""API for running simulations."""

import os
import json
import uuid
import gzip
from json import JSONDecodeError
import itertools
from typing import List, Dict, Callable, Union, Any, Optional, Tuple, Iterable
import datetime
import numpy as np
import pandas as pd
from panqec.codes import StabilizerCode
from panqec.decoders import BaseDecoder
from panqec.error_models import BaseErrorModel
from panqec.config import (
    CODES, ERROR_MODELS, DECODERS, PANQEC_DIR
)
from panqec.utils import identity, NumpyEncoder
from . import (
    BaseSimulation, DirectSimulation, SplittingSimulation
)


def run_once(
    code: StabilizerCode,
    error_model: BaseErrorModel,
    decoder: BaseDecoder,
    error_rate: float,
    rng=None
) -> dict:
    """Run a simulation once and return the results as a dictionary.

    Parameters
    ----------
    code : StabilizerCode
        The QEC code object to be run.
    error_model : BaseErrorModel
        The error model from which to sample errors from.
    decoder : BaseDecoder
        The decoder to use for correct the errors.
    rng :
        Numpy random number generator, used for deterministic seeding.

    Returns
    -------
    results : idct
        Results containing the following keys: error, syndrome, correction,
        effective_error, success, codespace
    """

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
    onefile: bool = True,
):
    """Run an input json file.

    Parameters
    ----------
    file_name : str
        Path to the input json file.
    n_trials : int
        The number of MCMC trials to do.
    start : Optional[int]
        If given, then instead of running all simulations in the sense of
        `simulations` only run `simulations[start:start + n_runs]`.
    n_runs : Optional[int]
        If given, then instead of running all simulations in the sense of
        `simulations` only run `simulations[start:start + n_runs]`.
    progress : Callable
        Callable function
    onefile : bool
        If set to True, then all outputs get written to one file.

    Returns
    -------
    None
    """
    print(f"Run file {file_name}")

    batch_sim = read_input_json(
        file_name, output_dir=output_dir,
        start=start, n_runs=n_runs,
        onefile=onefile
    )

    print("test")
    if verbose:
        print(f'running {len(batch_sim._simulations)} simulations:')
        for simulation in batch_sim._simulations:
            code = simulation.code.label
            noise = simulation.error_model.label
            if isinstance(simulation, SplittingSimulation):
                decoder = simulation.decoders[0].label
                error_rates = simulation.error_rates
                print(f'{code}, {noise}, {decoder}, {error_rates}')
            elif isinstance(simulation, DirectSimulation):
                decoder = simulation.decoder.label
                error_rate = simulation.error_rate
                print(f'{code}, {noise}, {decoder}, {error_rate}')
    batch_sim.run(n_trials, progress=progress)


class Simulation:
    """Quantum Error Correction Simulation.

    Parameters
    ----------
    code : StabilizerCode
        The QEC code object to be run.
    error_model : BaseErrorModel
        The error model from which to sample errors from.
    decoder : BaseDecoder
        The decoder to use for correct the errors.
    error_rate : float
        The error probability that denotes severity of noise.
    rng :
        Numpy random number generator, used for deterministic seeding.
    compress : bool
        Will compress results in .json.gz files if True.
        True by default.
    """

    start_time: datetime.datetime
    code: StabilizerCode
    error_model: BaseErrorModel
    decoder: BaseDecoder
    error_rate: float
    label: str
    _results: dict = {}
    rng = None
    compress: bool

    def __init__(
        self,
        code: StabilizerCode,
        error_model: BaseErrorModel,
        decoder: BaseDecoder,
        error_rate: float, rng=None,
        compress: bool = True,
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
        self.compress = compress

    @property
    def wall_time(self):
        """total amount of time spent on this simulation."""
        return self._results['wall_time']

    def run(self, repeats: int):
        """Run assuming perfect measurement.

        Parameters
        ----------
        repeats : int
            Number of times to run.
        """
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
        """Number of runs with results simulated already."""
        return len(self._results['success'])

    @property
    def results(self):
        """Raw results."""
        res = self._results
        return res

    @property
    def file_name(self) -> str:
        """Name of results file to save to.
        Not the full path.
        """
        if self.compress:
            extension = '.json.gz'
        else:
            extension = '.json'
        file_name = self.label + extension
        return file_name

    def get_file_path(self, output_dir: str) -> str:
        """Get the file path to save to."""
        file_path = os.path.join(output_dir, self.file_name)
        return file_path

    def load_results(self, output_dir: str):
        """Load previously written results from directory.

        Parameters
        ----------
        output_dir : str
            Path to the output directory where previous output files were
            saved.
        """
        file_path = self.get_file_path(output_dir)

        # Find the alternative compressed file path if it doesn't exist.
        if not os.path.exists(file_path):
            alt_file_path = file_path
            if os.path.splitext(file_path)[-1] == '.json':
                alt_file_path = file_path + '.gz'
            elif os.path.splitext(file_path)[-1] == '.gz':
                alt_file_path = file_path.replace('.json.gz', '.json')
            if os.path.exists(alt_file_path):
                file_path = alt_file_path
        try:
            if os.path.exists(file_path):
                if os.path.splitext(file_path)[-1] == '.gz':
                    with gzip.open(file_path, 'rb') as gz:
                        data = json.loads(gz.read().decode('utf-8'))
                else:
                    with open(file_path) as json_file:
                        data = json.load(json_file)
                self.load_results_from_dict(data)
        except JSONDecodeError as err:
            print(f'Error loading existing results file {file_path}')
            print('Starting this from scratch')
            print(err)

    def load_results_from_dict(self, data):
        """Directly load results from a parsed output file.

        Parameters
        ----------
        data : Dict
            The results that have been parsed from a previously saved output
            file.
        """
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

    def get_results_to_save(self) -> dict:
        """Get the results to save as a dict.

        Processed in a way to have a consistent format.
        """
        data = {
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
        }
        return data

    def save_results(self, output_dir: str):
        """Save results to directory.

        Parameters
        ----------
        output_dir : str
            The directory to save output files to.
        """
        data = self.get_results_to_save()
        if self.compress:
            with gzip.open(self.get_file_path(output_dir), 'wb') as gz:
                gz.write(json.dumps(data, cls=NumpyEncoder).encode('utf-8'))
        else:
            with open(self.get_file_path(output_dir), 'w') as json_file:
                json.dump(data, json_file, indent=4, cls=NumpyEncoder)

    def get_results(self):
        """Return results as dictionary that might be useful for later
        analysis.

        Returns
        -------
        simulation_data : Dict[str, Any]
            Data including both inputs and results.
        """

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
    """Holds and controls many simulations.

    Parameters
    ----------
    label : str
        The label of the files.
    on_update : Callable
        Function that gets called on every update.
    update_frequency : int
        Frequency at which to update results.
    save_frequency : int
        Frequency at which to write results to file on disk.
    output_dir : Optional[str]
        The directory to output to.
    onefile : bool
        Saves to one combined file if True, as opposed to a lot of smaller
        files.
    """

    _simulations: List[BaseSimulation]
    update_frequency: int
    save_frequency: int
    _output_dir: str
    onefile: bool

    def __init__(
        self,
        label='unlabelled',
        on_update: Callable = identity,
        update_frequency: int = 5,
        save_frequency: int = 1,
        output_dir: Optional[str] = None,
        method: str = "direct",
        verbose: bool = True,
        onefile: bool = False,
    ):
        self._simulations = []
        self.code: Dict = {}
        self.decoder: Dict = {}
        self.update_frequency = update_frequency
        self.save_frequency = save_frequency
        self.label = label
        self.method = method
        self.verbose = verbose
        if output_dir is not None:
            if onefile:
                self._output_dir = output_dir
            else:
                self._output_dir = os.path.join(output_dir, self.label)
        else:
            self._output_dir = os.path.join(PANQEC_DIR, self.label)
        os.makedirs(self._output_dir, exist_ok=True)
        self.onefile = onefile

    def __getitem__(self, *args):
        return self._simulations.__getitem__(*args)

    def __iter__(self):
        return self._simulations.__iter__()

    def __next__(self):
        return self._simulations.__next__()

    def __len__(self):
        return self._simulations.__len__()

    def append(self, simulation: BaseSimulation):
        """Append a simulation to the current BatchSimulation.

        Parameters
        ----------
        simulation: BaseSimulation
            The simulation to append.
        """
        self._simulations.append(simulation)

    def load_results(self):
        """Load results from disk."""
        for simulation in self._simulations:
            simulation.load_results(self._output_dir)

    def on_update(self):
        """Function that gets called on every update."""
        pass

    def estimate_remaining_time(self, n_trials: int):
        """Estimate remaining time given target n_trials.

        Parameters
        ----------
        n_trials : int
            The target total number of trials.
        """
        return sum(
            (n_trials - sim.n_results)*sim.wall_time/sim.n_results
            for sim in self._simulations
            if sim.n_results != 0
        )

    @property
    def wall_time(self):
        """Total time run so far."""
        return sum(sim.wall_time for sim in self._simulations)

    def run(self, n_trials, progress: Callable = identity):
        """Perform the running.

        Parameters
        ----------
        n_trials : int
            Number of trials to run.
        progress : Callable
            The progress bar, such as tqdm.
        """
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

        # for simulation in self._simulations:
        #     if self.verbose:
        #         print(f"\nPost-processing {simulation.label}")
        #     simulation.postprocess()

    def _save_results(self):
        for i_simulation, simulation in enumerate(self._simulations):
            if self.onefile:
                self._update_onefile(
                    i_simulation, simulation.get_results_to_save()
                )
            else:
                simulation.save_results(self._output_dir)

    def get_onefile_path(self) -> str:
        """Get path of combined all-in-one output file."""
        return os.path.join(self._output_dir, self.label + '.json.gz')

    def save_onefile(self):
        """Do a complete save of the onefile."""
        out_file = self.get_onefile_path()
        combined_data = []
        for simulation in self._simulations:
            combined_data.append(simulation.get_results_to_save())
        with gzip.open(out_file, 'wb') as gz:
            gz.write(
                json.dumps(combined_data, cls=NumpyEncoder).encode('utf-8')
            )

    def _update_onefile(self, i_simulation: int, new_data: dict) -> None:
        """Update only the i-th simulation's results to the onefile."""
        out_file = self.get_onefile_path()

        # First time file does not exist, so write it ot start.
        if not os.path.isfile(out_file):
            self.save_onefile()

        # A bit slow, but just unzip the previously existing zip file.
        with gzip.open(out_file, 'rb') as gz:
            combined_data = json.loads(gz.read().decode('utf-8'))

        # Update only the i-th simulation with the new data.
        combined_data[i_simulation] = new_data

        # Write the updated list to the .json.gz file.
        with gzip.open(out_file, 'wb') as gz:
            gz.write(
                json.dumps(combined_data, cls=NumpyEncoder).encode('utf-8')
            )

    def load_onefile(self):
        pass

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

    def get_results_df(self):
        batch_results = self.get_results()
        # print(
        #     'wall_time =',
        #     str(datetime.timedelta(seconds=batch_sim.wall_time))
        # )
        # print('n_trials = ', min(sim.n_results for sim in batch_sim))
        for sim, batch_result in zip(self, batch_results):
            n_logicals = batch_result['k']

            # Small fix for the current situation. TO REMOVE in later versions
            if n_logicals == -1:
                n_logicals = 1

            batch_result['noise_direction'] = sim.error_model.direction

            if self.method == 'direct':
                if len(sim.results['effective_error']) > 0:
                    batch_result['p_x'] = np.array(
                        sim.results['effective_error']
                    )[:, :n_logicals].any(axis=1).mean()
                    batch_result['p_x_se'] = np.sqrt(
                        batch_result['p_x']*(1 - batch_result['p_x'])
                        / (sim.n_results + 1)
                    )
                    batch_result['p_z'] = np.array(
                        sim.results['effective_error']
                    )[:, n_logicals:].any(axis=1).mean()
                    batch_result['p_z_se'] = np.sqrt(
                        batch_result['p_z']*(1 - batch_result['p_z'])
                        / (sim.n_results + 1)
                    )
                else:
                    batch_result['p_x'] = np.nan
                    batch_result['p_x_se'] = np.nan
                    batch_result['p_z'] = np.nan
                    batch_result['p_z_se'] = np.nan

        results = batch_results

        results_df = pd.DataFrame(results)

        if self.method == 'splitting':
            results_df = results_df.explode(['error_rates', 'p_est', 'p_se'])

        return results_df


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
        code = code_class(**code_params)  # type: ignore
    else:
        code = code_class(*code_params)  # type: ignore
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

    decoder = decoder_class(**decoder_params)
    return decoder


def read_input_json(file_path: str, *args, **kwargs) -> BatchSimulation:
    """Read json input file or .json.gz file."""
    try:
        if os.path.splitext(file_path)[-1] == '.json':
            with open(file_path) as f:
                data = json.load(f)
        else:
            with gzip.open(file_path, 'rb') as g:
                data = json.loads(g.read().decode('utf-8'))
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
    data: dict, start: Optional[int] = None, n_runs: Optional[int] = None,
    verbose: bool = True
) -> List[BaseSimulation]:
    simulations: List[BaseSimulation] = []

    method = 'direct'

    method_params = {}

    # The case if the ranges attribute is a list of dicts.
    # This allows everything to be jammed into one input file.
    if 'ranges' in data and isinstance(data['ranges'], list):
        for sub_ranges in data['ranges']:
            sub_data = dict(data)
            sub_data['ranges'] = sub_ranges
            simulations += get_simulations(sub_data)
        return simulations

    if 'ranges' in data:
        (
            code_range, noise_range, decoder_range, error_rates
        ) = _parse_all_ranges(data['ranges'])

        codes = [_parse_code_dict(code_dict) for code_dict in code_range]
        error_models = [_parse_error_model_dict(noise_dict)
                        for noise_dict in noise_range]
        instances: Iterable[Tuple] = itertools.product(
            codes, error_models, decoder_range, error_rates
        )

        if 'method' in data['ranges']:
            method = data['ranges']['method']['name']
            method_params = data['ranges']['method']['parameters']

    elif 'runs' in data:
        codes = [_parse_code_dict(run['code']) for run in data['runs']]
        decoder_range = [run['decoder'] for run in data['runs']]
        error_models = [_parse_error_model_dict(run['noise'])
                        for run in data['runs']]
        error_rates = [run['probability'] for run in data['runs']]
        instances = zip(codes, error_models, decoder_range, error_rates)

    else:
        raise ValueError("Invalid data format: does not have 'runs'\
                         or 'ranges' key")

    if method == 'direct':
        for code, error_model, decoder_dict, error_rate in instances:
            decoder = _parse_decoder_dict(decoder_dict, code, error_model,
                                          error_rate)

            simulations.append(DirectSimulation(code, error_model, decoder,
                                                error_rate, verbose=verbose,
                                                **method_params))

    if method == 'splitting':
        for code, error_model, decoder_dict in instances:
            decoders = [_parse_decoder_dict(decoder_dict, code, error_model, p)
                        for p in error_rates]

            simulations.append(SplittingSimulation(
                code, error_model, decoders, error_rates,
                verbose=verbose, **method_params
            ))

    if start is not None:
        simulations = simulations[start:]
    if n_runs is not None:
        simulations = simulations[:n_runs]

    return simulations


def read_input_dict(
    data: dict,
    start: Optional[int] = None,
    n_runs: Optional[int] = None,
    verbose: bool = True,
    *args, **kwargs
) -> BatchSimulation:
    """Return BatchSimulation from input dict.

    Parameters
    ----------
    data : dict
        Data that has been parsed from an input json file.
    start : Optional[int]
        Restrict the BatchSimulation skip this number of simulations.
        This is useful if you want a batch simulation with
        `simulations[start:start + n_runs]`.
    n_runs : Optional[int]
        Return this many simulations in the returned BatchSimulation.

    Returns
    -------
    batch_simulation : BatchSimulation
        BatchSimulation object populated with the simulations specified in the
        input data.
    """
    label = 'unlabelled'
    method = 'direct'

    if 'ranges' in data:

        # If many ranges are given label is 'combined', but if there is
        # only one unique label, then we can just use that label.
        if isinstance(data['ranges'], list):
            label = 'combined'
            labels = []
            for subdata in data['ranges']:
                if 'ranges' in subdata:
                    if 'label' in subdata['ranges']:
                        labels.append(subdata['ranges']['label'])
            if labels:
                unique_labels = list(set(labels))
                if len(unique_labels) == 1:
                    label = unique_labels[0]
        elif 'label' in data['ranges']:
            label = data['ranges']['label']

        if 'method' in data['ranges']:
            method = data['ranges']['method']['name']

    # If writing to one combined .json.gz file for everything, then the
    # label should be a random UUID.
    if 'onefile' in kwargs and kwargs['onefile']:
        label = str(uuid.uuid4().hex)

    kwargs['label'] = label
    kwargs['method'] = method
    kwargs['verbose'] = verbose

    print("Start batch simulation instance")
    batch_sim = BatchSimulation(*args, **kwargs)
    assert len(batch_sim._simulations) == 0

    simulations = get_simulations(data, start=start, n_runs=n_runs,
                                  verbose=verbose)

    for sim in simulations:
        batch_sim.append(sim)

    return batch_sim


def merge_results_dicts(results_dicts: List[Dict]) -> Dict:
    """Merge results dicts into one dict.

    The input attribute must be the same for each dict in the list.

    Parameters
    ----------
    results_dicts : List[Dict]
        List of data dicts to merge into one.

    Returns
    -------
    merged_dict : Dict
        Dictionary that contains the merged data.

    """

    # If splitting method
    if 'log_p_errors' in results_dicts[0]['results'].keys():
        error_rates = results_dicts[0]['results']['error_rates']
        results: Dict[str, Any] = {
            'log_p_errors': [[] for _ in error_rates],
            'n_runs': 0,
            'wall_time': 0.0
        }
        inputs = results_dicts[0]['inputs']
        for results_dict in results_dicts:
            assert results_dict['inputs'] == inputs, (
                'Warning: attempting to merge results of different inputs'
            )
            results['n_runs'] += results_dict['results']['n_runs']
            results['wall_time'] += results_dict['results']['wall_time']

            for i, p in enumerate(results_dict['results']['log_p_errors']):
                results['log_p_errors'][i].append(p)

        results['error_rates'] = error_rates

    # If direct method
    else:
        results = {
            'effective_error': [],
            'success': [],
            'codespace': [],
            'wall_time': 0.0
        }
        # For backward compatibilities (when n_runs was not in results)
        if 'n_runs' in results_dicts[0]['results'].keys():
            results['n_runs'] = 0
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


def merge_lists_of_results_dicts(
    results_dicts: List[Union[dict, list]]
) -> List[dict]:
    """List of results lists of dicts produced by BatchSimulation onefile.

    A more general version of `merge_results_dicts()` that can allow merging
    lists of results dicts that may not necessarily have the same inputs.

    Parameters
    ----------
    results_dicts : List[Union[dict, list]]
        List of dicts or list of lists of dicts that are to be merge.

    Returns
    -------
    combined_results : List[dict]
        The combined results as a list of dicts.
    """
    flattened_results_dicts = []
    for element in results_dicts:
        if isinstance(element, dict):
            flattened_results_dicts.append(element)
        elif isinstance(element, list):
            for value in element:
                flattened_results_dicts.append(value)

    # Combine results by sorting unique inputs.
    input_jsons = [
        json.dumps(element['inputs'])
        for element in flattened_results_dicts
    ]

    # The combined results is a list.
    combined_results = []
    for unique_input in set(input_jsons):
        combined_results.append(merge_results_dicts([
            flattened_results_dicts[i]
            for i, value in enumerate(input_jsons)
            if value == unique_input
        ]))
    return combined_results
