from typing import Tuple
from abc import ABCMeta, abstractmethod
import numpy as np
from panqec.codes import StabilizerCode


class BaseErrorModel(metaclass=ABCMeta):
    """Base class for error models"""

    @property
    @abstractmethod
    def label(self) -> str:
        """Label used in plots result files
        E.g. 'PauliErrorModel X1 Y0 Z0'
        """

    @property
    def id(self) -> str:
        return self.__class__.__name__

    @property
    @abstractmethod
    def params(self) -> dict:
        """List of class arguments (as a dictionary), that can be saved
        and reused to instantiate the same error model"""

    @abstractmethod
    def generate(
        self, code: StabilizerCode, error_rate: float, rng=None
    ) -> np.ndarray:
        """Generate errors for a given code and probability of failure

        Parameters
        ----------
        code : StabilizerCode
            Errors will be generated on the qubits of the provided code
        error_rate: float
            Physical error rate
        rng: numpy.random.Generator
            Random number generator (default=None resolves to
            numpy.random.default_rng())

        Returns
        -------
        error : np.ndarray
            Error as an array of size 2n (with n the number of qubits)
            in the binary symplectic format
        """

    @abstractmethod
    def probability_distribution(
        self, code: StabilizerCode, error_rate: float
    ) -> Tuple:
        """Probability distribution of X, Y and Z errors on all the
        qubits of a code. Can be used to generate errors and configure
        decoders.

        Parameters
        ----------
        code : StabilizerCode
            Code used for the error model
        error_rate: float
            Physical error rate

        Returns
        -------
        p_i, p_x, p_y, p_z : Tuple[np.ndarray]
            Probability distribution for I, X, Y and Z errors.
            Each probability is an array of size n (number of qubits)
        """

    def error_probability(
        self, error: np.ndarray, code: StabilizerCode,
        error_rate: float, log_output: bool = False
    ) -> np.ndarray:

        pi, px, py, pz = self.probability_distribution(code, error_rate)

        prob_vector = np.zeros(code.n)
        prob_vector += py * (error[:code.n] == error[code.n:])
        prob_vector += px * np.logical_and(error[:code.n],
                                           np.logical_not(error[code.n:]))
        prob_vector += pz * np.logical_and(np.logical_not(error[:code.n]),
                                           error[code.n:])
        prob_vector += pi * np.logical_and(np.logical_not(error[:code.n]),
                                           np.logical_not(error[code.n:]))

        if log_output:
            prob = np.sum(np.log(prob_vector))
        else:
            prob = np.prod(prob_vector)

        return prob

    def get_weights(
        self, code: StabilizerCode, error_rate: float, eps=1e-20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get MWPM weights for deformed Pauli noise."""

        pi, px, py, pz = self.probability_distribution(
            code, error_rate
        )

        total_p_x = px + py
        total_p_z = pz + py

        weights_x = -np.log(
            (total_p_x + eps) / (1 - total_p_x + eps)
        )
        weights_z = -np.log(
            (total_p_z + eps) / (1 - total_p_z + eps)
        )

        return weights_x, weights_z
