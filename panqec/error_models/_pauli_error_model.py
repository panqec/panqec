import functools
from typing import Tuple, Optional
import numpy as np
from panqec.codes import StabilizerCode
from . import BaseErrorModel
from panqec.bpauli import pauli_to_bsf
import random


def fast_choice(options, probs, rng=None):
    """Found on stack overflow to accelerate np.random.choice"""
    if rng is None:
        x = random.random()
    else:
        x = rng.random()
    cum = 0
    for i, p in enumerate(probs):
        cum += p
        if x < cum:
            return options[i]
    return options[-1]


class PauliErrorModel(BaseErrorModel):
    """Pauli channel IID noise model.

    The overal quantum channel takes the form

    .. math::

        \\mathcal{\\rho}(\\rho) =
        (1 - p)\\rho + p(r_X X \\rho X + r_Y Y \\rho Y + r_Z Z \\rho Z)

    where it is required that :math:`r_X + r_Y + r_Z =1`.
    """

    def __init__(
        self,
        r_x: float, r_y: float, r_z: float,
        deformation_name: Optional[str] = None,
        deformation_kwargs: Optional[dict] = None
    ):
        """Initialize Pauli error model at a given rate of X, Y and Z errors,
        i.e. :math:`p_\\sigma = p r_\\sigma` for
        :math:`\\sigma \\in \\{X, Y, Z\\}`,
        :math:`p` the total error rate,
        and :math:`p_\\sigma` the probability of getting
        the error :math:`\\sigma` on each qubit.

        Parameters
        ----------
        r_x : float
            Rate of X errors
        r_y : float
            Rate of Y errors
        r_z : float
            Rate of Z errors
        deformation_name : str, optional
            Name of the Clifford deformation to apply to the noise model.
            The Clifford deformation must be provided in the code class.
        """
        if not np.isclose(r_x + r_y + r_z, 1):
            raise ValueError(
                f'Noise direction ({r_x}, {r_y}, {r_z}) does not sum to 1.0'
            )
        self._direction = r_x, r_y, r_z
        self._deformation_name = deformation_name

        if deformation_kwargs is not None:
            self._deformation_kwargs = deformation_kwargs
        else:
            self._deformation_kwargs = {}

    @property
    def direction(self) -> Tuple[float, float, float]:
        """Rate of X, Y and Z errors, as given when initializing the
        error model

        Returns
        -------
        (r_x, r_y, r_z): Tuple[float]
            Rate of X, Y and Z errors
        """
        return self._direction

    @property
    def label(self):
        label = 'Pauli X{:.4f}Y{:.4f}Z{:.4f}'.format(*self.direction)
        if self._deformation_name:
            label = 'Deformed ' + self._deformation_name + ' ' + label

        return label

    @property
    def params(self) -> dict:
        """List of class arguments (as a dictionary), that can be saved
        and reused to instantiate the same code"""
        return {
            'r_x': self.direction[0],
            'r_y': self.direction[1],
            'r_z': self.direction[2],
            'deformation_name': self._deformation_name,
            'deformation_kwargs': self._deformation_kwargs
        }

    def generate(self, code: StabilizerCode, error_rate: float, rng=None):
        rng = np.random.default_rng() if rng is None else rng

        p_i, p_x, p_y, p_z = self.probability_distribution(code, error_rate)

        error_pauli = ''.join([fast_choice(
            ('I', 'X', 'Y', 'Z'),
            [p_i[i], p_x[i], p_y[i], p_z[i]],
            rng=rng
        ) for i in range(code.n)])

        error = pauli_to_bsf(error_pauli)

        return error

    @functools.lru_cache()
    def probability_distribution(
        self, code: StabilizerCode, error_rate: float
    ) -> Tuple:
        n = code.n
        r_x, r_y, r_z = self.direction

        p: dict = {}
        p['I'] = (1 - error_rate) * np.ones(n)
        p['X'] = (r_x * error_rate) * np.ones(n)
        p['Y'] = (r_y * error_rate) * np.ones(n)
        p['Z'] = (r_z * error_rate) * np.ones(n)

        if self._deformation_name is not None:
            for i in range(code.n):
                deformation = code.get_deformation(
                    code.qubit_coordinates[i], self._deformation_name,
                    **self._deformation_kwargs
                )
                previous_p = {pauli: p[pauli][i] for pauli in ['X', 'Y', 'Z']}
                for pauli in ['X', 'Y', 'Z']:
                    p[pauli][i] = previous_p[deformation[pauli]]

        return p['I'], p['X'], p['Y'], p['Z']
