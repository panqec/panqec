from typing import Tuple
from ._rotated_planar_3d_pauli import RotatedPlanar3DPauli


class LayeredToricPauli(RotatedPlanar3DPauli):

    def vertex(self, operator: str, location: Tuple[int, int, int]):
        pass

    def face(self, operator: str, location: Tuple[int, int, int]):
        pass
