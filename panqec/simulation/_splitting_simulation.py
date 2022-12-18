"""
API for running simulations.
"""
import numpy as np
import datetime
from typing import List, Tuple
from panqec.codes import StabilizerCode
from panqec.decoders import BaseDecoder
from panqec.error_models import BaseErrorModel
from . import calculate_logical_error_rate, BaseSimulation


def g(x):
    return 1 / (1 + x)


g = np.vectorize(g)


class SplittingSimulation(BaseSimulation):
    """Quantum Error Correction Simulation."""

    start_time: datetime.datetime
    code: StabilizerCode
    error_model: BaseErrorModel
    decoder: BaseDecoder
    error_rates: np.ndarray
    current_error: List[np.ndarray]  # error vector for each error rate
    label: str
    _results: dict = {}
    rng = None

    def __init__(
        self,
        code: StabilizerCode,
        error_model: BaseErrorModel,
        decoders: List[BaseDecoder],
        error_rates: List[float],
        n_init_runs: int,
        start_run: int = 0,
        compress: bool = True,
        verbose: bool = True,
        rng=None
    ):
        super().__init__(
            code, error_model, compress=compress, verbose=verbose, rng=rng
        )

        self.decoders = decoders
        self.error_rates = np.sort(error_rates)[::-1]
        self.n_init_runs = n_init_runs

        self.current_error = []
        self.initial_logical_p = None
        self.start_run = start_run

        self._results = {
            **self._results,
            'error_rates': self.error_rates,
            'log_p_errors': [[] for _ in self.error_rates],
            'logical_error_rates': []
        }
        self._inputs = {
            **self._inputs,
            'decoder': {
                'name': self.decoders[0].id,
                'parameters': self.decoders[0].params,
            },
            'error_rates': self.error_rates,
            'method': {
                'name': 'splitting',
                'parameters': {
                    'n_init_runs': n_init_runs,
                    'start_run': start_run
                }
            }
        }

    def _run(self, n_runs: int):
        """Run assuming perfect measurement."""
        # Find an error that leads to a decoding failure

        if len(self.current_error) == 0:
            # # Only works for deformed 3D rotated toric code
            # # TODO; remove that (or replace it)
            # initial_error = np.concatenate([np.zeros(self.code.n),
            #                                 np.ones(self.code.n)])
            # deformation_indices = self.error_model.get_deformation_indices(
            #     self.code
            # )
            # initial_error[self.code.n:][deformation_indices] = 0
            # initial_error[:self.code.n][deformation_indices] = 1
            if (self.error_model.error_probability(
                self.code.logicals_x[0], self.code, 0.5
            ) != 0):
                initial_error: np.ndarray = self.code.logicals_x[0]
            elif (self.error_model.error_probability(
                self.code.logicals_z[0], self.code, 0.5
            ) != 0):
                initial_error = self.code.logicals_z[0]
            else:
                raise NotImplementedError(
                    "Splitting method: neither of the logicals has nonzero"
                    "probability. Please specify another error that fails"
                    "when decoded"
                )

            # Check that the chosen error indeed fails
            syndrome = self.code.measure_syndrome(initial_error)
            correction = self.decoders[0].decode(syndrome)
            total_error = (correction + initial_error) % 2
            if self.code.is_success(total_error):
                raise ValueError(
                    "Splitting method: the chosen initial error"
                    "does not fail when decoded"
                )
            self.current_error = [initial_error
                                  for _ in range(len(self.error_rates))]

        for i_run in range(n_runs):
            for i_p, error_rate in enumerate(self.error_rates):
                self.current_error[i_p], log_p_error = self.get_next_error(
                    self.decoders[i_p], error_rate, self.current_error[i_p]
                )
                self._results['log_p_errors'][i_p].append(log_p_error)
            self._results['n_runs'] += 1

    def postprocess(self):
        super().postprocess()

        logical_p = self.compute_logical_probabilities()
        self._results['logical_error_rates'] = logical_p

    def get_results(self):
        """Return results as dictionary."""

        # self.postprocess()

        simulation_data = {
            'size': self.code.size,
            'code': self.code.label,
            'n': self.code.n,
            'k': self.code.k,
            'd': self.code.d,
            'error_model': self.error_model.label,
            'error_rates': self.error_rates,
            'n_runs': self._results['n_runs'],
            'p_est': self._results['logical_error_rates']
        }

        # Use posterior Beta distribution of the effective error rate
        # standard distribution as standard error.
        # TODO: change that to the actual splitting method std
        simulation_data['p_se'] = np.sqrt(
            simulation_data['p_est']*(1 - simulation_data['p_est'])
            / (simulation_data['n_runs'] + 1)
        )
        return simulation_data

    def get_next_error(
        self,
        decoder: BaseDecoder,
        error_rate: float,
        previous_error: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not (0 <= error_rate <= 1):
            raise ValueError('Error rate must be in [0, 1].')

        new_edge = np.zeros(2*self.code.n, dtype='uint')
        e_index = np.random.choice(self.code.n)

        pi, px, py, pz = self.error_model.probability_distribution(self.code,
                                                                   error_rate)

        paulis = []
        if px[e_index] != 0:
            paulis.append('X')
        if py[e_index] != 0:
            paulis.append('Y')
        if pz[e_index] != 0:
            paulis.append('Z')

        e_pauli = np.random.choice(paulis)

        if e_pauli == 'X' or e_pauli == 'Y':
            new_edge[e_index] = 1
        if e_pauli == 'Z' or e_pauli == 'Y':
            new_edge[self.code.n + e_index] = 1

        new_error = (previous_error + new_edge) % 2

        log_p_previous_error = self.error_model.error_probability(
            previous_error, self.code, error_rate, log_output=True
        )
        log_p_new_error = self.error_model.error_probability(
            new_error, self.code, error_rate, log_output=True
        )

        q = np.exp(min(0, log_p_new_error - log_p_previous_error))
        b = np.random.choice([0, 1], p=[1-q, q])

        next_error = previous_error
        log_p_next_error = log_p_previous_error

        if b:
            syndrome = self.code.measure_syndrome(new_error)
            correction = decoder.decode(syndrome)
            total_error = (correction + new_error) % 2
            if (self.code.is_logical_error(total_error)
                    or not self.code.in_codespace(total_error)):
                next_error = new_error
                log_p_next_error = log_p_new_error

        return next_error, log_p_next_error

    def compute_optimal_c(self):
        list_c = np.linspace(0.0001, 1, 100)
        n_p = len(self.error_rates)
        log_p_errors = np.array(self._results['log_p_errors'])

        optimal_c = []

        for j in range(n_p-1):
            print(f"{j+1} / {n_p - 1}")
            lhs = []
            rhs = []
            for c in list_c:
                print(f"C : {c}", end="\r")
                lhs.append(np.sum(
                    g(c * np.exp(log_p_errors[j][self.start_run:]
                      - log_p_errors[j+1][self.start_run:]))
                ))
                rhs.append(np.sum(
                    g(1/c * np.exp(log_p_errors[j+1][self.start_run:]
                      - log_p_errors[j][self.start_run:]))
                ))

            lhs = np.array(lhs)
            rhs = np.array(rhs)

            try:
                idx = np.argwhere(np.diff(np.sign(lhs - rhs))).flatten()[0]
                optimal_c.append(list_c[idx])
            except IndexError:
                optimal_c.append(1)

        return optimal_c

    def compute_logical_probabilities(self):
        n_p = len(self.error_rates)
        log_p_errors = np.array(self._results['log_p_errors'])

        logical_p = np.zeros(n_p)

        if self.verbose:
            print("Compute initial logical error rate")

        self.initial_logical_p = calculate_logical_error_rate(
            self.code, self.error_model, self.decoders[0],
            self.error_rates[0], self.n_init_runs, verbose=True
        )
        logical_p[0] = self.initial_logical_p

        if self.verbose:
            print(f"P = {self.initial_logical_p}")

        if self.verbose:
            print("Compute optimal C")
        optimal_c = self.compute_optimal_c()

        if self.verbose:
            print("Compute logical error rates")

        for j in range(n_p-1):
            c = optimal_c[j]
            numerator = np.sum(
                g(c * np.exp(log_p_errors[j][self.start_run:]
                             - log_p_errors[j+1][self.start_run:]))
            )
            denominator = np.sum(
                g(1/c * np.exp(log_p_errors[j+1][self.start_run:]
                               - log_p_errors[j][self.start_run:]))
            )
            ratio = c * numerator / denominator

            logical_p[j+1] = logical_p[j] * ratio

        return logical_p
