import itertools
from typing import Optional
import numpy as np
from typing import Dict
from qecsim.model import StabilizerCode
from ..tc3d import Toric3DPauli


class DeformableToric3DPauli(Toric3DPauli):
    """Pauli Operator in 3D Toric Code lattice, but deformable."""

    _deform_map: Dict[str, str] = {
        'I': 'I',
        'X': 'Z',
        'Y': 'Y',
        'Z': 'X'
    }

    def __init__(self, code: StabilizerCode, bsf: Optional[np.ndarray] = None):
        if bsf is not None:
            bsf = bsf.copy()
        super(DeformableToric3DPauli, self).__init__(code, bsf=bsf)

    def deform(self):

        # The axis edge to deform operators on.
        deform_axis = self.code.X_AXIS

        # The ranges of indices to iterate over.
        ranges = [range(length) for length in self.code.shape]

        # Change the Xs to Zs and Zs to Xs on the edges to deform.
        for axis, x, y, z in itertools.product(*ranges):
            original_operator = self.operator((axis, x, y, z))
            deformed_operator = self._deform_map[original_operator]

            # Only deform on the axis to be deformed.
            if axis == deform_axis:

                # Apply the operator at the site again to remove it.
                self.site(original_operator, (axis, x, y, z))

                # Apply the new operator at the site.
                self.site(deformed_operator, (axis, x, y, z))
