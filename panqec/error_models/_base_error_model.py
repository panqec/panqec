from typing import Tuple
from abc import ABCMeta, abstractmethod
import numpy as np
from panqec.codes import StabilizerCode
from scipy.sparse import csr_matrix


class BaseErrorModel(metaclass=ABCMeta):
    """Base class for error models"""

    @property
    @abstractmethod
    def label(self):
        """Label used in plots result files
        E.g. 'PauliErrorModel X1 Y0 Z0'
        """

    @abstractmethod
    def generate(self, code: StabilizerCode, probability: float, rng=None) -> csr_matrix:
        """Generate errors for a given code and probability of failure

        Parameters
        ----------
        code : StabilizerCode
            Errors will be generated on the qubits of the provided code
        probability: float
            Physical error rate
        rng: numpy.random.Generator
            Random number generator (default=None resolves to numpy.random.default_rng())

        Returns
        -------
        error : scipy.sparse.csr_matrix
            Error as a sparse array of size 1 x 2n (with n the number of qubits)
            in the binary symplectic format
        """

    @abstractmethod
    def probability_distribution(
        self, code: StabilizerCode, probability: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Probability distribution of X, Y and Z errors on all the qubits of a code
        Can be used to generate errors and configure decoders

        Parameters
        ----------
        code : StabilizerCode
            Code used for the error model
        probability: float
            Physical error rate

        Returns
        -------
        p_i, p_x, p_y, p_z : Tuple[np.ndarray]
            Probability distribution for I, X, Y and Z errors.
            Each probability is an array of size n (number of qubits)
        """
