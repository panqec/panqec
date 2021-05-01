import numpy as np
from qecsim.model import ErrorModel
from qecsim.model import StabilizerCode
from ..noise import PauliErrorModel
from ._deformable_toric_3d_pauli import DeformableToric3DPauli


class DeformedPauliErrorModel(ErrorModel):

    _undeformed_model: PauliErrorModel

    def __init__(self, r_x, r_y, r_z):
        self._undeformed_model = PauliErrorModel(r_x, r_y, r_z)

    @property
    def label(self) -> str:
        return 'Deformed Pauli {!r}'.format(self.direction)

    def generate(
        self, code: StabilizerCode, probability: float, rng=None
    ) -> np.ndarray:
        error = self._undeformed_model.generate(code, probability, rng)
        deformed_error = self._deform_operator(code, error)
        return deformed_error

    def _deform_operator(
        self, code: StabilizerCode, error: np.ndarray
    ) -> np.ndarray:
        pauli = DeformableToric3DPauli(code, bsf=error)
        pauli.deform()
        return pauli.to_bsf()
