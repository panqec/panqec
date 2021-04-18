from qecsim.models.toric import ToricPauli
from ._toric_code_3d import ToricCode3D


class Toric3DPauli(ToricPauli):
    """Pauli Operator on 3D Toric Code."""

    def __init__(self, code: ToricCode3D, bsf=None):
        super().__init__(code, bsf)
