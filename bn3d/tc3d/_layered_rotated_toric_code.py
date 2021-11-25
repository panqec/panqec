from typing import Optional
from qecsim.model import StabilizerCode


class LayeredRotatedToricCode(StabilizerCode):
    """Layered Rotated Code for good subthreshold scaling."""

    def __init__(
        self, L_x: int,
        L_y: Optional[int] = None,
        L_z: Optional[int] = None
    ):
        if L_y is None:
            L_y = L_x
        if L_z is None:
            L_z = L_x

        self._size = (L_x, L_y, L_z)
        self._qubit_index = self._create_qubit_indices()
        self._vertex_index = self._create_vertex_indices()
        self._face_index = self._create_face_indices()
