"""
API for running simulations.
"""
from typing import List
import datetime
import numpy as np
from qecsim.model import StabilizerCode, ErrorModel, Decoder


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
    results: list = []

    def __init__(self, code, error_model, decoder):
        self.code = code
        self.error_model = error_model
        self.decoder = decoder

    def run(self, repeats: int):
        pass


class BatchSimulation():

    simulations: List[Simulation]
    n_trials: int

    def __init__(self):
        pass
