from typing import Tuple, Dict
from bn3d.models import StabilizerCode

Indexer = Dict[Tuple[int, int], int]  # coordinate to index


class Planar3DCode(StabilizerCode):
    @property
    def dimension(self) -> int:
        return 3

    @property
    def label(self) -> str:
        return 'Planar {}x{}x{}'.format(*self.size)

    def _vertex(self, location: Tuple[int, int, int], deformed_axis: int = None) -> Dict[str, Tuple]:
        x, y, z = location

        if (x, y, z) not in self.vertex_index:
            raise ValueError(f"Invalid coordinate {location} for a vertex")

        pauli = 'Z'
        deformed_pauli = 'X'

        delta = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

        operator = dict()
        for d in delta:
            qubit_location = (x + d[0], y + d[1], z + d[2])

            if self.is_qubit(qubit_location):
                is_deformed = (self.axis(qubit_location) == deformed_axis)
                operator[qubit_location] = deformed_pauli if is_deformed else pauli

        return operator

    def _face(self, location: Tuple[int, int, int], deformed_axis: int = None) -> Dict[str, Tuple]:
        x, y, z = location

        if (x, y, z) not in self.face_index:
            raise ValueError(f"Invalid coordinate {location} for a face")

        pauli = 'X'
        deformed_pauli = 'Z'

        # Face in xy-plane.
        if z % 2 == 0:
            delta = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0)]
        # Face in yz-plane.
        elif (x % 2 == 0):
            delta = [(0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
        # Face in zx-plane.
        elif (y % 2 == 0):
            delta = [(-1, 0, 0), (1, 0, 0), (0, 0, -1), (0, 0, 1)]

        operator = dict()
        for d in delta:
            qubit_location = (x + d[0], y + d[1], z + d[2])

            if self.is_qubit(qubit_location):
                is_deformed = (self.axis(qubit_location) == deformed_axis)
                operator[qubit_location] = deformed_pauli if is_deformed else pauli

        return operator

    def axis(self, location):
        x, y, z = location

        if (z % 2 == 0) and (x % 2 == 1) and (y % 2 == 0):
            axis = self.X_AXIS
        elif (z % 2 == 0) and (x % 2 == 0) and (y % 2 == 1):
            axis = self.Y_AXIS
        elif (z % 2 == 1) and (x % 2 == 0) and (y % 2 == 0):
            axis = self.Z_AXIS
        else:
            raise ValueError(f'Location {location} does not correspond to a qubit')

        return axis

    def _get_qubit_coordinates(self) -> Indexer:
        coordinates = []
        Lx, Ly, Lz = self.size

        # Qubits along e_x
        for x in range(1, 2*Lx+1, 2):
            for y in range(0, 2*Ly, 2):
                for z in range(0, 2*Lz, 2):
                    coordinates.append((x, y, z))

        # Qubits along e_y
        for x in range(2, 2*Lx, 2):
            for y in range(1, 2*Ly-1, 2):
                for z in range(0, 2*Lz, 2):
                    coordinates.append((x, y, z))

        # Qubits along e_z
        for x in range(2, 2*Lx, 2):
            for y in range(0, 2*Ly, 2):
                for z in range(1, 2*Lz-1, 2):
                    coordinates.append((x, y, z))

        coord_to_index = {coord: i for i, coord in enumerate(coordinates)}

        return coord_to_index

    def _get_vertex_coordinates(self) -> Indexer:
        coordinates = []
        Lx, Ly, Lz = self.size

        for x in range(2, 2*Lx, 2):
            for y in range(0, 2*Ly, 2):
                for z in range(0, 2*Lz, 2):
                    coordinates.append((x, y, z))

        coord_to_index = {coord: i for i, coord in enumerate(coordinates)}

        return coord_to_index

    def _get_face_coordinates(self) -> Indexer:
        coordinates = []
        Lx, Ly, Lz = self.size

        # Face in xy plane
        for x in range(1, 2*Lx+1, 2):
            for y in range(1, 2*Ly-1, 2):
                for z in range(0, 2*Lz, 2):
                    coordinates.append((x, y, z))

        # Face in yz plane
        for x in range(2, 2*Lx, 2):
            for y in range(1, 2*Ly-1, 2):
                for z in range(1, 2*Lz, 2):
                    coordinates.append((x, y, z))

        # Face in xz plane
        for x in range(1, 2*Lx+1, 2):
            for y in range(0, 2*Ly, 2):
                for z in range(1, 2*Lz-1, 2):
                    coordinates.append((x, y, z))

        coord_to_index = {coord: i for i, coord in enumerate(coordinates)}

        return coord_to_index

    def _get_logicals_x(self) -> Dict[str, Tuple]:
        """The unique logical X operator."""

        Lx, Ly, Lz = self.size
        logicals = []

        # X operators along x edges in x direction.
        operator = dict()
        for x in range(1, 2*Lx+1, 2):
            operator[(x, 0, 0)] = 'X'
        logicals.append(operator)

        return logicals

    def _get_logicals_z(self) -> Dict[str, Tuple]:
        """The unique logical Z operator."""

        Lx, Ly, Lz = self.size
        logicals = []

        # X operators along x edges in x direction.
        operator = dict()
        for y in range(0, 2*Ly, 2):
            for z in range(0, 2*Lz, 2):
                operator[(1, y, z)] = 'Z'
        logicals.append(operator)

        return logicals
