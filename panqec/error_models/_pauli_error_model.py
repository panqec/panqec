import functools
from typing import Tuple
import numpy as np
from panqec.codes import StabilizerCode
from . import BaseErrorModel
from panqec.bpauli import pauli_to_bsf


class PauliErrorModel(BaseErrorModel):
    """Pauli channel IID noise model."""

    def __init__(self, r_x: float, r_y: float, r_z: float):
        """Initialize Pauli error model at a given rate of X, Y and Z errors,
        i.e. $P(u) = p * r_u$ for $u \in \{X, Y, Z\}$, $p$ the total error rate, and
        $P(u)$ the probability of getting the error $u$ on each qubit.

        Parameters
        ----------
        r_x : float
            Rate of X errors
        r_y : float
            Rate of Y errors
        r_z : float
            Rate of Z errors
        """
        if not np.isclose(r_x + r_y + r_z, 1):
            raise ValueError(
                f'Noise direction ({r_x}, {r_y}, {r_z}) does not sum to 1.0'
            )
        self._direction = r_x, r_y, r_z

    @property
    def direction(self) -> Tuple[float, float, float]:
        """Rate of X, Y and Z errors, as given when initializing the error model

        Returns
        -------
        (r_x, r_y, r_z): Tuple[float]
            Rate of X, Y and Z errors
        """
        return self._direction

    @property
    def label(self):
        return 'Pauli X{:.4f}Y{:.4f}Z{:.4f}'.format(*self.direction)

    def generate(self, code: StabilizerCode, probability: float, rng=None):
        rng = np.random.default_rng() if rng is None else rng

        p_i, p_x, p_y, p_z = self.probability_distribution(code, probability)

        error_pauli = ''.join([rng.choice(
            ('I', 'X', 'Y', 'Z'),
            p=[p_i[i], p_x[i], p_y[i], p_z[i]]
        ) for i in range(code.n)])

        error = pauli_to_bsf(error_pauli)

        return error

    @functools.lru_cache()
    def probability_distribution(
        self, code: StabilizerCode, probability: float
    ) -> Tuple:
        n = code.n
        r_x, r_y, r_z = self.direction

        p_i = (1 - probability) * np.ones(n)
        p_x = (r_x * probability) * np.ones(n)
        p_y = (r_y * probability) * np.ones(n)
        p_z = (r_z * probability) * np.ones(n)

        return p_i, p_x, p_y, p_z
