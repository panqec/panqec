import datetime
import numpy as np
from panqec.codes import FoliatedCode
from panqec.decoders import BaseDecoder
from panqec.error_models import BaseErrorModel
from panqec.simulation import BaseSimulation


class ClusterStateSimulation(BaseSimulation):
    """Quantum Error Correction Simulation."""

    start_time: datetime.datetime
    code: FoliatedCode
    error_model: BaseErrorModel
    decoder: BaseDecoder
    error_rate: float
    label: str
    _results: dict = {}
    rng = None

    def __init__(
        self,
        code: FoliatedCode,
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
            'codespace': []
        }
        self._inputs = {
            **self._inputs,
            'decoder': {
                'name': self.decoder.id,
                'parameters': self.decoder.params
            },
            'error_rate': self.error_rate,
            'method': {
                'name': 'cluster_state',
                'parameters': {}
            }
        }

    def _run(self, n_runs: int):
        """Run assuming perfect measurement."""

        results = self.code.run(n_runs, self.error_rate)

        for key, value in results.items():
            if key in self._results.keys():
                self._results[key] = results[key]

        self._results['n_runs'] += n_runs

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
                simulation_data['n_fail'] / simulation_data['n_runs']
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
