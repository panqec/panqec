from typing import Tuple
from bn3d.models import StabilizerPauli
import numpy as np


class RotatedToric3DPauli(StabilizerPauli):

    def on_defect_boundary(self, Lx, Ly, x, y):
        """Determine whether or not to defect each boundary."""
        defect_x_boundary = False
        defect_y_boundary = False
        if Lx % 2 == 1 and x == 2*Lx-1:
            defect_x_boundary = True
        if Ly % 2 == 1 and y == 2*Ly-1:
            defect_y_boundary = True
        return defect_x_boundary, defect_y_boundary

    def vertex(self, operator: str, location: Tuple[int, int, int], deformed_axis=None):
        Lx, Ly, Lz = self.code.size
        x, y, z = location

        defect_x_boundary, defect_y_boundary = self.on_defect_boundary(Lx, Ly, x, y)
        defect_operator = {'X': 'Z', 'Y': 'Y', 'Z': 'X'}[operator]

        deformed_operator = {'X': 'Z', 'Z': 'X'}[operator]
        deformed_defect_operator = {'X': 'Z', 'Z': 'X'}[defect_operator]

        delta = [(1, -1, 0), (-1, 1, 0), (1, 1, 0), (-1, -1, 0), (0, 0, 1), (0, 0, -1)]

        for d in delta:
            qx, qy, qz = tuple(np.add(location, d))

            qubit_location = (qx % (2*Lx), qy % (2*Ly), qz)

            if self.code.is_qubit(qubit_location):
                defect_x_on_edge = defect_x_boundary and qubit_location[0] == 0
                defect_y_on_edge = defect_y_boundary and qubit_location[1] == 0
                is_deformed = (self.code.axis(qubit_location) == deformed_axis)

                if defect_x_on_edge != defect_y_on_edge:
                    if is_deformed:
                        self.site(deformed_defect_operator, qubit_location)
                    else:
                        self.site(defect_operator, qubit_location)
                else:
                    if is_deformed:
                        self.site(deformed_operator, qubit_location)
                    else:
                        self.site(operator, qubit_location)

    def face(self, operator: str, location: Tuple[int, int, int], deformed_axis=None):
        Lx, Ly, Lz = self.code.size
        x, y, z = location

        defect_x_boundary, defect_y_boundary = self.on_defect_boundary(
            Lx, Ly, x, y
        )
        defect_operator = {'X': 'Z', 'Y': 'Y', 'Z': 'X'}[operator]

        deformed_operator = {'X': 'Z', 'Z': 'X'}[operator]
        deformed_defect_operator = {'X': 'Z', 'Z': 'X'}[defect_operator]

        # z-normal so face is xy-plane.
        if z % 2 == 1:
            delta = [(-1, -1, 0), (1, 1, 0), (-1, 1, 0), (1, -1, 0)]
        # x-normal so face is in yz-plane.
        elif (x + y) % 4 == 2:
            delta = [(-1, -1, 0), (1, 1, 0), (0, 0, -1), (0, 0, 1)]
        # y-normal so face is in zx-plane.
        elif (x + y) % 4 == 0:
            delta = [(-1, 1, 0), (1, -1, 0), (0, 0, -1), (0, 0, 1)]

        for d in delta:
            qx, qy, qz = tuple(np.add(location, d))

            qubit_location = (qx % (2*Lx), qy % (2*Ly), qz)

            if self.code.is_qubit(qubit_location):
                defect_x_on_edge = defect_x_boundary and qubit_location[0] == 0
                defect_y_on_edge = defect_y_boundary and qubit_location[1] == 0
                is_deformed = (self.code.axis(qubit_location) == deformed_axis)

                if defect_x_on_edge != defect_y_on_edge:
                    if is_deformed:
                        self.site(deformed_defect_operator, qubit_location)
                    else:
                        self.site(defect_operator, qubit_location)
                else:
                    if is_deformed:
                        self.site(deformed_operator, qubit_location)
                    else:
                        self.site(operator, qubit_location)
