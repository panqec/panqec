import functools
from typing import Tuple
import numpy as np
from qecsim import paulitools as pt
from qecsim.models.generic import SimpleErrorModel
from panqec.models import StabilizerCode


class PauliErrorModel(SimpleErrorModel):
    """Pauli channel IID noise model."""

    # direction: Tuple[float, float, float]

    def __init__(self, r_x, r_y, r_z):
        if not np.isclose(r_x + r_y + r_z, 1):
            raise ValueError(
                f'Noise direction ({r_x}, {r_y}, {r_z}) does not sum to 1.0'
            )
        self._direction = r_x, r_y, r_z

    @property
    def direction(self):
        return self._direction

    @property
    def label(self):
        return 'Pauli X{:.4f}Y{:.4f}Z{:.4f}'.format(*self.direction)

    def generate(self, code: StabilizerCode, probability: float, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        n_qubits = code.n
        p_i, p_x, p_y, p_z = self.probability_distribution(code, probability)

        error_pauli = ''.join([rng.choice(
            ('I', 'X', 'Y', 'Z'),
            p=[p_i[i], p_x[i], p_y[i], p_z[i]]
        ) for i in range(n_qubits)])

        bsf = pt.pauli_to_bsf(error_pauli)

        return bsf

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