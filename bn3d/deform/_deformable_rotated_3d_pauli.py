import itertools
from typing import Optional
import numpy as np
from typing import Dict
from qecsim.model import StabilizerCode
from ..tc3d import Rotated3DPauli


class DeformableRotated3DPauli(Rotated3DPauli):
    """Pauli Operator in 3D Toric Code lattice, but deformable."""

    # Mapping from undeformed Pauli operator to deformed Pauli operator.
    _deform_map: Dict[str, str] = {
        'I': 'I',
        'X': 'X',
        'Y': 'Z',
        'Z': 'Y'
    }

    def __init__(self, code: StabilizerCode, bsf: Optional[np.ndarray] = None):
        super(DeformableRotated3DPauli, self).__init__(code, bsf=bsf)

    def deform(self):
        """Replace Z with Y on every horizontal Z stabilizer"""

        # Change the Xs to Zs and Zs to Xs on the edges to deform.
        for coord in self.code.qubit_index.keys():
            x, y, z = coord
            original_operator = self.operator(coord)
            deformed_operator = self._deform_map[original_operator]

            # Only deform on the axis to be deformed.
            if z % 2 == 1:
                # Apply the operator at the site again to remove it.
                self.site(original_operator, coord)

                # Apply the new operator at the site.
                self.site(deformed_operator, coord)
