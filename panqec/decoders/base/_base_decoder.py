from abc import ABCMeta, abstractmethod
from typing import Optional, List
from panqec.codes import StabilizerCode
from panqec.error_models import BaseErrorModel
import numpy as np


class BaseDecoder(metaclass=ABCMeta):
    """Base class for decoders"""

    def __init__(
        self,
        code: StabilizerCode,
        error_model: BaseErrorModel,
        error_rate: float
    ):
        self.code = code
        self.error_model = error_model
        self.error_rate = error_rate

    @property
    @abstractmethod
    def allowed_codes(self) -> Optional[List[str]]:
        """List of codes concerned by the decoders (by id).
        Takes the value None if all codes are allowed.
        Example: ['Toric2DCode', 'Planar2DCode', 'RotatedPlanar2DCode']
        """

    @property
    @abstractmethod
    def label(self):
        """Label used in plots and result files
        E.g. 'Toric 2D Matching'
        """

    @abstractmethod
    def decode(self, syndrome: np.ndarray, **kwargs) -> np.ndarray:
        """Given a code and a syndrome, returns a correction to apply
        to the qubits

        Parameters
        ----------
        syndrome: np.ndarray
            Syndrome as an array of size m, where m is the number of
            stabilizers. Each element contains 1 if the stabilizer is
            activated and 0 otherwise

        kwargs: dict
            Decoder-specific parameters (implemented by subclasses)

        Returns
        -------
        correction : np.ndarray
            Correction as an array of size 2n (with n the number of qubits)
            in the binary symplectic format.
        """
