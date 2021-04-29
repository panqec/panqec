import itertools
import numpy as np
from qecsim.model import ErrorModel
from qecsim.model import StabilizerCode
from ..noise import PauliErrorModel
from ..tc3d import Toric3DPauli


class DeformedPauliErrorModel(ErrorModel):

    _undeformed_model: PauliErrorModel

    def __init__(self):
        self._undeformed_model = PauliErrorModel()

    @property
    def label(self) -> str:
        return 'Deformed Pauli {!r}'.format(self.direction)

    def generate(
        self, code: StabilizerCode, probability: float, rng=None
    ) -> np.ndarray:
        error = self._undeformed_model.generate(code, probability, rng)
        deformed_error = deform_operator(code, error)
        return deformed_error


def deform_operator(code: StabilizerCode, error: np.ndarray) -> np.ndarray:
    pauli = Toric3DPauli(code, bsf=error)
    ranges = [range(length) for length in code.size]

    # The axis edge to deform operators on.
    axis = code.X_AXIS

    # Change the Xs to Zs and Zs to Xs on the edges to deform.
    for L_x, L_y, L_z in itertools.product(*ranges):
        if pauli.operator((axis, L_x, L_y, L_z)) == 'X':
            pauli.site('Y', axis, L_x, L_y, L_z)
        elif pauli.operator((axis, L_x, L_y, L_z)) == 'Z':
            pauli.site('Y', axis, L_x, L_y, L_z)

    return pauli.to_bsf()
