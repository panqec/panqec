import numpy as np
from ._toric_code_3d import ToricCode3D


class Toric3DPauli():
    """Pauli Operator on 3D Toric Code."""

    code: ToricCode3D
    _bsf: np.ndarray
    _xs: np.ndarray
    _zs: np.ndarray

    def __init__(self, code: ToricCode3D, bsf=None):
        self.code = code
        if bsf is None:
            self._bsf = np.zeros(code.shape, dtype=np.uint)
        else:
            self.bsf = bsf

    def site(self, operator: str, *indices: tuple):
        """Apply Pauli at a site or many sites."""
        for index in indices:

            # Index is modulo shape of code.
            index = tuple(np.mod(index, self.code.shape))

            # Flip sites.
            if operator in ('X', 'Y'):
                self._xs[index] ^= 1
            if operator in ('Z', 'Y'):
                self._zs[index] ^= 1

    def to_bsf(self) -> np.ndarray:
        """Get binary simplectic form."""
        return np.concatenate((self._xs.flatten(), self._zs.flatten()))
