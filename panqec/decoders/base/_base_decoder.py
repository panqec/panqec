from abc import ABCMeta, abstractmethod
from panqec.codes import StabilizerCode
import numpy as np


class BaseDecoder(metaclass=ABCMeta):
    """Base class for decoders"""

    def __init__(self, error_model, probability):
        self._error_model = error_model
        self._probability = probability

    @property
    @abstractmethod
    def label(self):
        """Label used in plots and result files
        E.g. 'Toric 3D Matching'
        """

    @abstractmethod
    def decode(
        self, code: StabilizerCode, syndrome: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Given code and syndrome, return correction to apply to the qubits

        Parameters
        ----------
        code : StabilizerCode
            Stabilizer code used by the decoder
        syndrome: scipy.sparse.np.ndarray
            Syndrome as a 1 x m sparse matrix (where m is the number of
            stabilizers).
            Each element contains 1 if the stabilizer is activated and 0
            otherwise
        kwargs: dict
            Decoder-specific parameters (implemented by subclasses)

        Returns
        -------
        correction : scipy.sparse.np.ndarray
            Correction as a sparse array of size 1 x 2n (with n the number of
            qubits)
            in the binary symplectic format.
        """
