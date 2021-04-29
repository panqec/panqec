from typing import Dict
import itertools
import numpy as np
from qecsim.model import ErrorModel
from qecsim.model import StabilizerCode
from ..noise import PauliErrorModel
from ..tc3d import Toric3DPauli


class DeformedPauliErrorModel(ErrorModel):

    _undeformed_model: PauliErrorModel
    _deform_map: Dict[str, str] = {
        'I': 'I',
        'X': 'Z',
        'Y': 'Y',
        'Z': 'X'
    }

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

        # The axis edge to deform operators on.
        deform_axis = code.X_AXIS
        original = Toric3DPauli(code, bsf=error)

        # Create a Toric3DPauli object.
        new_pauli = Toric3DPauli(code)

        # The ranges of indices to iterate over.
        ranges = [range(length) for length in code.shape]

        # Change the Xs to Zs and Zs to Xs on the edges to deform.
        for axis, x, y, z in itertools.product(*ranges):
            original_operator = original.operator((axis, x, y, z))
            deformed_operator = self._deform_map[original_operator]
            if axis == deform_axis:
                new_pauli.site(deformed_operator, (axis, x, y, z))
            else:
                new_pauli.site(original_operator, (axis, x, y, z))

        return new_pauli.to_bsf()
