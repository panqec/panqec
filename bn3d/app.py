"""
API for running simulations.
"""
import json
import itertools
from typing import List, Dict
import datetime
import numpy as np
from qecsim.model import StabilizerCode, ErrorModel, Decoder
from .config import codes, error_models, decoders, noise_dependent_decoders


def run_once(code, error_model, decoder, error_probability, rng=None):

    if not (0 <= error_probability <= 1):
        raise ValueError('Error probability must be in [0, 1].')

    if rng is None:
        rng = np.random.default_rng()


class Simulation:

    start_time: datetime.datetime
    code: StabilizerCode
    error_model: ErrorModel
    decoder: Decoder
    error_probability: float
    results: list = []

    def __init__(self, code, error_model, decoder, probability):
        self.code = code
        self.error_model = error_model
        self.decoder = decoder

    def run(self, repeats: int):
        run_once(
            self.code, self.error_model, self.decoder,
            error_probability=self.error_probability
        )


class BatchSimulation():

    _simulations: List[Simulation]
    n_trials: int

    def __init__(self):
        self._simulations = []

    def append(self, simulation: Simulation):
        self._simulations.append(simulation)


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
    """Parse a single run dict."""
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
    if 'runs' in data:
        runs = data['runs']
    if 'ranges' in data:
        runs += expand_inputs_ranges(data['ranges'])

    batch_sim = BatchSimulation()

    for run in runs:
        simulation = parse_run(run)
        batch_sim.append(simulation)

    return batch_sim
