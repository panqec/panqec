"""
API for running simulations.
"""

import datetime
import numpy as np
from panqec.codes import StabilizerCode
from panqec.decoders import BaseDecoder
from panqec.error_models import BaseErrorModel
from ..bpauli import get_effective_error
from . import BaseSimulation


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
    effective_error = get_effective_error(
        total_error, code.logicals_x, code.logicals_z
    )
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


def calculate_logical_error_rate(
    code: StabilizerCode,
    error_model: BaseErrorModel,
    decoder: BaseDecoder,
    error_rate: float,
    n_runs: int,
    verbose: bool = False
):
    """Simple function to calculate the logical error rate"""

    n_fails = 0
    for run in range(n_runs):
        if verbose:
            print(f"Run {run+1} / {n_runs}", end='\r')
        results = run_once(code, error_model, decoder, error_rate)
        n_fails += 1 - results['success']

    return n_fails / n_runs


class DirectSimulation(BaseSimulation):
    """Quantum Error Correction Simulation.

    Parameters
    -----------
    code : StabilizerCode
        The code to simulate.
    error_model : BaseErrorModel
        The error model to use.
    decoder: BaseDecoder
        The decoder to use.
    error_rate : float
        The error rate parameter.
    compress : bool
        Set False to not compress the output files and save as plain json.
    verbose : bool
        Set False to suppress output.
    rng :
        Set Random number generator if you want to seed it.
    """

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
        error_rate: float,
        compress: bool = True,
        verbose=True,
        rng=None
    ):
        super().__init__(
            code, error_model, compress=compress, verbose=verbose, rng=rng
        )

        self.decoder = decoder
        self.error_rate = error_rate

        self._results = {
            **self._results,
            'effective_error': [],
            'success': [],
            'codespace': [],
        }
        self._inputs = {
            **self._inputs,
            'decoder': {
                'name': self.decoder.id,
                'parameters': self.decoder.params
            },
            'error_rate': self.error_rate,
            'method': {
                'name': 'direct',
                'parameters': {}
            }
        }

    def _run(self, n_runs: int):
        """Run assuming perfect measurement."""

        for i_run in range(n_runs):
            shot = run_once(
                self.code, self.error_model, self.decoder,
                error_rate=self.error_rate,
                rng=self.rng
            )
            for key, value in shot.items():
                if key in self._results.keys():
                    self._results[key].append(value)

            self._results['n_runs'] += 1

    def get_results(self):
        """Return results as dictionary."""

        success = np.array(self.results['success'])
        if len(success) > 0:
            n_fail = np.sum(~success)
        else:
            n_fail = 0
        simulation_data = {
            'n_success': np.sum(success),
            'n_fail': n_fail,
            'n_runs': len(success),
        }

        # Use sample mean as estimator for effective error rate.
        if simulation_data['n_runs'] != 0:
            simulation_data['p_est'] = (
                simulation_data['n_fail']/simulation_data['n_runs']
            )
        else:
            simulation_data['p_est'] = np.nan

        # Use posterior Beta distribution of the effective error rate
        # standard distribution as standard error.
        simulation_data['p_se'] = np.sqrt(
            simulation_data['p_est']*(1 - simulation_data['p_est'])
            / (simulation_data['n_runs'] + 1)
        )
        return simulation_data
