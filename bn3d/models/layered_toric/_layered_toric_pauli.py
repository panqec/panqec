from typing import Tuple
from ..rotated_planar_3d._rotated_planar_3d_pauli import RotatedPlanar3DPauli
import numpy as np


class LayeredToricPauli(RotatedPlanar3DPauli):

    def on_defect_boundary(self, Lx, Ly, x, y):
        """Determine whether or not to defect each boundary."""
        defect_x_boundary = False
        defect_y_boundary = False
        if Lx % 2 == 1 and x == 2*Lx:
            defect_x_boundary = True
        if Ly % 2 == 1 and y == 2*Ly:
            defect_y_boundary = True
        return defect_x_boundary, defect_y_boundary

    def vertex(self, operator: str, location: Tuple[int, int, int], deformed_axis=None):
        Lx, Ly, Lz = self.code.size
        x, y, z = location

        defect_x_boundary, defect_y_boundary = self.on_defect_boundary(Lx, Ly, x, y)

        if defect_x_boundary or defect_y_boundary:
            defect_operator = {'X': 'Z', 'Y': 'Y', 'Z': 'X'}[operator]

        delta = [(1, -1, 0), (-1, 1, 0), (1, 1, 0), (-1, -1, 0), (0, 0, 1), (0, 0, -1)]

        for d in delta:
            qx, qy, qz = tuple(np.add(location, d))
            qubit_location = (qx % (2*Lx), qy % (2*Lz), qz)

            if self.code.is_qubit(qubit_location):
                defect_x_on_edge = defect_x_boundary and qubit_location[0] == 1
                defect_y_on_edge = defect_y_boundary and qubit_location[1] == 1
                if defect_x_on_edge != defect_y_on_edge:
                    self.site(defect_operator, qubit_location)
                else:
                    self.site(operator, qubit_location)

    def face(self, operator: str, location: Tuple[int, int, int], deformed_axis=None):
        Lx, Ly, Lz = self.code.size
        x, y, z = location
        print("Face", location)

        defect_x_boundary, defect_y_boundary = self.on_defect_boundary(
            Lx, Ly, x, y
        )
        if defect_x_boundary or defect_y_boundary:
            defect_operator = {'X': 'Z', 'Y': 'Y', 'Z': 'X'}[operator]

        # z-normal so face is xy-plane.
        if z % 2 == 1:
            delta = [(-1, -1, 0), (1, 1, 0), (-1, 1, 0), (1, -1, 0)]
        # x-normal so face is in yz-plane.
        elif (x + y) % 4 == 0:
            delta = [(-1, -1, 0), (1, 1, 0), (0, 0, -1), (0, 0, 1)]
        # y-normal so face is in zx-plane.
        elif (x + y) % 4 == 2:
            delta = [(-1, 1, 0), (1, -1, 0), (0, 0, -1), (0, 0, 1)]

        for d in delta:
            qx, qy, qz = tuple(np.add(location, d))

            if (Ly % 2 == 1 and qy > 2*Ly) or Ly % 2 == 0 or qy < 0:
                qy %= 2*Ly

            if (Lx % 2 == 1 and qx > 2*Lx) or Lx % 2 == 0 or qx < 0:
                qx %= 2*Lx

            qubit_location = (qx, qy, qz)

            if self.code.is_qubit(qubit_location):
                defect_x_on_edge = defect_x_boundary and qubit_location[0] == 1
                defect_y_on_edge = defect_y_boundary and qubit_location[1] == 1
                if defect_x_on_edge != defect_y_on_edge:
                    self.site(defect_operator, qubit_location)
                else:
                    self.site(operator, qubit_location)
