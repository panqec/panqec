"""API for running simulations."""

import os
import json
from json import JSONDecodeError
import datetime
import itertools
from typing import List, Dict, Callable, Union, Any, Optional, Tuple, Iterable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from panqec.codes import StabilizerCode
from panqec.decoders import BaseDecoder
from panqec.error_models import BaseErrorModel
from panqec.config import (
    CODES, ERROR_MODELS, DECODERS
)
from panqec.utils import identity, load_json, save_json
from . import (
    BaseSimulation, DirectSimulation, SplittingSimulation
)
from panqec.analysis import Analysis


def run_file(
    input_file: str,
    output_file: str,
    n_trials: int,
    progress: Callable = identity,
    log_file: Optional[str] = None,
    verbose: bool = True,
):
    """Run an input json file.

    Parameters
    ----------
    input_file : str
        Path to the input json file.
    output_file: str
        Path to the json file that will contain the results.
    n_trials : int
        The number of MCMC trials to do.
    progress : Callable
        Callable function
    verbose: bool,
        Verbosity of the output
    Returns
    -------
    None
    """
    print(f"Run file {input_file}")

    batch_sim = read_input_json(input_file, output_file, log_file=log_file)

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


class BatchSimulation():
    """Holds and controls many simulations.

    Parameters
    ----------
    label : str
        The label of the files.
    output_file : str
        Path to the file (.json or .json.gz) that will store
        the simulation results
    label: str
        Label of the batch simulation
    on_update : Callable
        Function that gets called on every update.
    update_frequency : int
        Frequency at which to update results.
    save_frequency : int
        Frequency at which to write results to file on disk.
    method: str
        The method can be either "direct" or "splitting".
        The direct method samples independent errors at each iteration,
        while the splitting method (by Bravyi & Vargo) uses MCMC to
        determine the next error to sample.
        Ref: arXiv:1308.6270
    """

    _simulations: List[BaseSimulation]
    update_frequency: int
    save_frequency: int
    _output_file: str

    def __init__(
        self,
        output_file: str,
        label='unlabelled',
        on_update: Callable = identity,
        update_frequency: int = 5,
        save_frequency: int = 1,
        method: str = "direct",
        log_file: Optional[str] = None,
        verbose: bool = True,
    ):
        self._simulations = []
        self.code: Dict = {}
        self.decoder: Dict = {}
        self.update_frequency = update_frequency
        self.save_frequency = save_frequency
        self.label = label
        self.method = method
        self.verbose = verbose
        self._output_file = output_file
        self._log_file = log_file

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
            simulation.load_results(self._output_file)

    def on_update(self, n_trials: int):
        """Function that gets called on every update.
        It uses the total number of runs, `n_trials`, to estimate
        the remaining time
        """
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
        min_current_trial = min([
            simulation.n_results for simulation in self._simulations
        ])

        for i_trial in progress(list(range(min_current_trial, n_trials))):
            for simulation in self._simulations:
                if simulation.n_results < n_trials:
                    simulation.run(1)
            if i_trial > 0:
                if i_trial % self.update_frequency == 0:
                    self.on_update(n_trials)
                if i_trial % self.save_frequency == 0:
                    self.save_results()
            if i_trial == n_trials - 1:
                self.on_update(n_trials)
                self.save_results()

            self._log_progress(i_trial, n_trials)

        # for simulation in self._simulations:
        #     if self.verbose:
        #         print(f"\nPost-processing {simulation.label}")
        #     simulation.postprocess()

    def _save_results(self):
        self._update_file(
            self.get_results_to_save()
        )

    def save_file(self):
        """Do a complete save of the file."""
        combined_data = []
        for simulation in self._simulations:
            combined_data.append(simulation.get_results_to_save())

        # Create output directory if it does not exist yet
        output_dir = os.path.dirname(self._output_file)
        if output_dir == '':
            output_dir = os.getcwd()

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        save_json(combined_data, self._output_file)

    def _log_progress(self, i_trial, n_trials):
        if self._log_file is not None:
            with open(self._log_file, "w") as f:
                f.write(f"{i_trial+1}/{n_trials}")

    def _update_file(self, new_data: list) -> None:
        """Update only the results to one file."""

        # First time file does not exist, so write it to start.
        if not os.path.isfile(self._output_file):
            self.save_file()

        # Write the updated list to the .json or .json.gz file.
        save_json(new_data, self._output_file)

    def load_file(self):
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

    def get_results_to_save(self):
        results = []
        for simulation in self._simulations:
            simulation_data = simulation.get_results_to_save()
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

    def activate_live_update(self):
        self.on_update = self._update_plot

    def _update_plot(self, n_trials: int):
        import IPython

        plt.clf()

        remaining_time = self.estimate_remaining_time(n_trials)

        analysis = Analysis(self._output_file, verbose=False)
        analysis.plot_thresholds(
            include_threshold_estimate=False,
            include_main_title=False,
            include_sector_title=False
        )
        plt.title(
            f'Time remaining '
            f'{datetime.timedelta(seconds=int(remaining_time))}'
        )

        IPython.display.clear_output(wait=True)
        IPython.display.display(plt.gcf())


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
    if 'parameters' in data['error_model']:
        params_range = _parse_parameters_range(
            data['error_model']['parameters']
        )
        noise_range = []
        for params in params_range:
            noise_range.append(data['error_model'].copy())
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

    error_rate_range = _parse_parameters_range(data['error_rate'])

    return code_range, noise_range, decoder_range, error_rate_range


def expand_input_ranges(data: dict) -> List[Dict]:
    runs: List[Dict] = []

    (
        code_range, error_model_range, decoder_range, error_rate_range
    ) = _parse_all_ranges(data)

    for (
        error_model_param, decoder_param, error_rate, code_param
    ) in itertools.product(
        error_model_range, decoder_range, error_rate_range, code_range
    ):
        run = {
            k: v for k, v in data.items()
            if k not in ['code', 'error_model', 'decoder', 'error_rate']
        }
        for key in ['code', 'error_model', 'decoder']:
            run[key] = {
                k: v for k, v in data[key].items() if k != 'parameters'
            }
        run['code']['parameters'] = code_param['parameters']
        run['error_model']['parameters'] = error_model_param['parameters']
        run['decoder']['parameters'] = decoder_param['parameters']
        run['error_rate'] = error_rate

        runs.append(run)

    return runs


def _parse_code_dict(code_dict: Dict[str, Any]) -> StabilizerCode:
    code_name = code_dict['name']
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
    error_model_name = noise_dict['name']
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
    decoder_name = decoder_dict['name']
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


def read_input_json(
    input_file: str,
    output_file: str,
    log_file: Optional[str] = None
) -> BatchSimulation:
    """Read json input file or .json.gz file."""
    try:
        data = load_json(input_file)
    except JSONDecodeError as err:
        print(f'Error reading input file {input_file}')
        raise err

    return read_input_dict(data, output_file, log_file=log_file)


def get_runs(data: dict) -> List[dict]:
    """Get expanded runs from input dictionary."""

    runs = []
    if 'runs' in data:
        runs = data['runs']
    if 'ranges' in data:
        runs += expand_input_ranges(data['ranges'])

    return runs


def count_runs(file_path: str) -> Optional[int]:
    """Count the number of noise parameters in an input file.

    Return None if no noise parameters range given.
    """
    n_runs = None
    with open(file_path) as f:
        data = json.load(f)
    all_runs = get_runs(data)
    n_runs = len(all_runs)
    return n_runs


def get_simulations(data: dict, verbose: bool = True) -> List[BaseSimulation]:
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
        print("Run", data['runs'])
        codes = [_parse_code_dict(run['code']) for run in data['runs']]
        decoder_range = [run['decoder'] for run in data['runs']]
        error_models = [_parse_error_model_dict(run['error_model'])
                        for run in data['runs']]
        error_rates = [run['error_rate'] for run in data['runs']]
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

    return simulations


def read_input_dict(
    data: dict,
    output_file: str,
    verbose: bool = True,
    *args, **kwargs
) -> BatchSimulation:
    """Return BatchSimulation from input dict.

    Parameters
    ----------
    data : dict
        Data that has been parsed from an input json file.
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

    kwargs['label'] = label
    kwargs['method'] = method
    kwargs['verbose'] = verbose

    print("Start batch simulation instance")
    batch_sim = BatchSimulation(output_file, *args, **kwargs)
    assert len(batch_sim._simulations) == 0

    simulations = get_simulations(data, verbose=verbose)

    for sim in simulations:
        batch_sim.append(sim)

    return batch_sim
