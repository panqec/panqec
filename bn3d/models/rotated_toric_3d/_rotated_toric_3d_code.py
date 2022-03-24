from typing import Tuple, Dict
import numpy as np
from bn3d.models import StabilizerCode

Indexer = Dict[Tuple[int, int, int], int]  # coordinate to index


def on_defect_boundary(Lx, Ly, x, y):
    """Determine whether or not to defect each boundary."""
    defect_x_boundary = False
    defect_y_boundary = False
    if Lx % 2 == 1 and x == 2*Lx-1:
        defect_x_boundary = True
    if Ly % 2 == 1 and y == 2*Ly-1:
        defect_y_boundary = True
    return defect_x_boundary, defect_y_boundary


class RotatedToric3DCode(StabilizerCode):
    """Rotated Toric Code for good subthreshold scaling."""

    @property
    def label(self) -> str:
        return 'Rotated Toric 3D {}x{}x{}'.format(*self.size)

    @property
    def dimension(self) -> int:
        return 3

    def _vertex(self, location: Tuple[int, int, int], deformed_axis: int = None) -> Dict[str, Tuple]:
        Lx, Ly, Lz = self.size
        x, y, z = location

        if location not in self.vertex_index:
            raise ValueError(f"Invalid coordinate {location} for a vertex")

        pauli = 'Z'
        deformed_pauli = 'X'

        defect_x_boundary, defect_y_boundary = on_defect_boundary(Lx, Ly, x, y)

        delta = [(1, -1, 0), (-1, 1, 0), (1, 1, 0), (-1, -1, 0), (0, 0, 1), (0, 0, -1)]

        operator = dict()
        for d in delta:
            qx, qy, qz = tuple(np.add(location, d))
            qubit_location = (qx % (2*Lx), qy % (2*Ly), qz)

            if self.is_qubit(qubit_location):
                defect_x_on_edge = defect_x_boundary and qubit_location[0] == 0
                defect_y_on_edge = defect_y_boundary and qubit_location[1] == 0
                is_deformed = (self.axis(qubit_location) == deformed_axis)
                has_defect = (defect_x_on_edge != defect_y_on_edge)

                operator[qubit_location] = deformed_pauli if is_deformed != has_defect else pauli

        return operator

    def _face(self, location: Tuple[int, int, int], deformed_axis: int = None) -> Dict[str, Tuple]:
        Lx, Ly, Lz = self.size
        x, y, z = location

        if location not in self.face_index:
            raise ValueError(f"Invalid coordinate {location} for a face")

        pauli = 'X'
        deformed_pauli = 'Z'

        defect_x_boundary, defect_y_boundary = on_defect_boundary(Lx, Ly, x, y)

        # z-normal so face is xy-plane.
        if z % 2 == 1:
            delta = [(-1, -1, 0), (1, 1, 0), (-1, 1, 0), (1, -1, 0)]
        # x-normal so face is in yz-plane.
        elif (x + y) % 4 == 2:
            delta = [(-1, -1, 0), (1, 1, 0), (0, 0, -1), (0, 0, 1)]
        # y-normal so face is in zx-plane.
        elif (x + y) % 4 == 0:
            delta = [(-1, 1, 0), (1, -1, 0), (0, 0, -1), (0, 0, 1)]

        operator = dict()
        for d in delta:
            qx, qy, qz = tuple(np.add(location, d))
            qubit_location = (qx % (2*Lx), qy % (2*Ly), qz)

            if self.is_qubit(qubit_location):
                defect_x_on_edge = defect_x_boundary and qubit_location[0] == 0
                defect_y_on_edge = defect_y_boundary and qubit_location[1] == 0
                is_deformed = (self.axis(qubit_location) == deformed_axis)
                has_defect = (defect_x_on_edge != defect_y_on_edge)

                operator[qubit_location] = deformed_pauli if is_deformed != has_defect else pauli

        return operator

    def axis(self, location: Tuple[int, int, int]) -> int:
        x, y, z = location

        if location not in self.qubit_index:
            raise ValueError(f'Location {location} does not correspond to a qubit')

        if z % 2 == 0:
            axis = self.Z_AXIS
        elif (x + y) % 4 == 0:
            axis = self.X_AXIS
        elif (x + y) % 4 == 2:
            axis = self.Y_AXIS

        return axis

    def _get_qubit_coordinates(self) -> Indexer:
        Lx, Ly, Lz = self.size

        coordinates = []

        # Horizontal
        for x in range(0, 2*Lx, 2):
            for y in range(0, 2*Ly, 2):
                for z in range(1, 2*Lz, 2):
                    coordinates.append((x, y, z))

        # Vertical
        for x in range(1, 2*Lx, 2):
            for y in range(1, 2*Ly, 2):
                for z in range(2, 2*Lz, 2):
                    if (x + y) % 4 == 0:
                        coordinates.append((x, y, z))

        coord_to_index = {coord: i for i, coord in enumerate(coordinates)}

        return coord_to_index

    def _get_vertex_coordinates(self) -> Indexer:
        Lx, Ly, Lz = self.size

        coordinates = []

        for x in range(1, 2*Lx, 2):
            for y in range(1, 2*Ly, 2):
                for z in range(1, 2*Lz, 2):
                    if (x + y) % 4 == 0:
                        coordinates.append((x, y, z))

        coord_to_index = {coord: i for i, coord in enumerate(coordinates)}

        return coord_to_index

    def _get_face_coordinates(self) -> Indexer:
        Lx, Ly, Lz = self.size

        coordinates = []

        # Horizontal faces
        for x in range(1, 2*Lx + 1, 2):
            for y in range(1, 2*Ly + 1, 2):
                for z in range(1, 2*Lz, 2):
                    if (x + y) % 4 == 2:
                        coordinates.append((x, y, z))

        # Vertical faces
        for x in range(0, 2*Lx, 2):
            for y in range(0, 2*Ly, 2):
                for z in range(2, 2*Lz, 2):
                    if not ((Lx % 2 == 0 and y == 0) or (Ly % 2 == 0 and x == 0)):
                        coordinates.append((x, y, z))

        coord_to_index = {coord: i for i, coord in enumerate(coordinates)}

        return coord_to_index

    def _get_logicals_x(self) -> np.ndarray:
        """Get the logical X operators."""

        Lx, Ly, Lz = self.size
        logicals = []

        # Even times even.
        if Lx % 2 == 0 and Ly % 2 == 0:
            # X string operator along y.
            operator = dict()
            for x, y, z in self.qubit_index:
                if y == 0 and z == 1:
                    operator[(x, y, z)] = 'X'
            logicals.append(operator)

            # X string operator along x.
            for x, y, z in self.qubit_index:
                if x == 0 and z == 1:
                    operator[(x, y, z)] = 'X'
            logicals.append(operator)

        # Odd times odd
        elif Lx % 2 == 1 and Ly % 2 == 1:
            operator = dict()
            for x, y, z in self.qubit_index:
                # X string operator in undeformed code. (OK)
                if z == 1 and x + y == 2*Lx-2:
                    operator[(x, y, z)] = 'X'
            logicals.append(operator)

        # Odd times even
        else:
            operator = dict()
            for x, y, z in self.qubit_index:
                # X string operator in undeformed code. (OK)
                if Lx % 2 == 1:
                    if z == 1 and x == 0:
                        operator[(x, y, z)] = 'X'
                else:
                    if z == 1 and y == 0:
                        operator[(x, y, z)] = 'X'

            logicals.append(operator)

        return logicals

    def _get_logicals_z(self) -> np.ndarray:
        """Get the logical Z operators."""

        Lx, Ly, Lz = self.size
        logicals = []

        # Even times even.
        if (Lx % 2 == 0) and (Ly % 2 == 0):
            operator = dict()
            for x, y, z in self.qubit_index:
                if x == 0:
                    operator[(x, y, z)] = 'Z'
            logicals.append(operator)

            operator = dict()
            for x, y, z in self.qubit_index:
                if y == 0:
                    operator[(x, y, z)] = 'Z'

            logicals.append(operator)

        # Odd times odd
        elif (Lx % 2 == 1) and (Ly % 2 == 1):
            operator = dict()
            for x, y, z in self.qubit_index:
                if x == y:
                    operator[(x, y, z)] = 'Z'

            logicals.append(operator)

        # Odd times even
        else:
            operator = dict()
            for x, y, z in self.qubit_index:
                if (Lx % 2 == 1 and y == 0) or (Ly % 2 == 1 and x == 0):
                    operator[(x, y, z)] = 'Y'

            logicals.append(operator)

        return logicals
