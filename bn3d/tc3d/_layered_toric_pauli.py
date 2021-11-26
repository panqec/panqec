from typing import Tuple
from ._rotated_planar_3d_pauli import RotatedPlanar3DPauli


class LayeredToricPauli(RotatedPlanar3DPauli):

    def vertex(self, operator: str, location: Tuple[int, int, int]):
        L_x, L_y, L_z = self.code.size
        x, y, z = location
        neighbours = []

        # Edges along unrotated x with periodic boundaries.
        neighbours += [
            ((x + 1) % (2*L_x), y - 1, z),
            (x - 1, (y + 1) % (2*L_y), z)
        ]

        # Edges along unrotated y with periodic boundaries.
        neighbours += [
            ((x + 1) % (2*L_x), (y + 1) % (2*L_y), z),
            (x - 1, y - 1, z)
        ]

        # Vertical edges along unrotated z, with smooth boundaries.
        if z + 1 <= 2*L_z + 1:
            neighbours.append((x, y, z + 1))
        if z - 1 >= 1:
            neighbours.append((x, y, z - 1))

        for edge in neighbours:
            self.site(operator, edge)

    def face(self, operator: str, location: Tuple[int, int, int]):
        pass
