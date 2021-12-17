from typing import Tuple
import numpy as np
from ..generic._indexed_code import IndexedCode
from ._rotated_planar_3d_pauli import RotatedPlanar3DPauli


class RotatedPlanarCode3D(IndexedCode):

    pauli_class = RotatedPlanar3DPauli

    @property
    def n_k_d(self) -> Tuple[int, int, int]:
        Lx, Ly, Lz = self.size
        n_horizontals = (2*Lx + 1) * (2*Ly + 1) * (Lz+1)
        n_verticals = (Lx + 1) * 2*Ly * Lz
        return (n_horizontals + n_verticals, 1, 2*min(Lx, Ly, Lz) + 1)

    @property
    def label(self) -> str:
        return 'Rotated Planar {}x{}x{}'.format(*self.size)

    @property
    def logical_xs(self) -> np.ndarray:
        """Get the unique logical X operator."""

        if self._logical_xs.size == 0:
            Lx, Ly, Lz = self.size
            logicals = []

            # X operators along x edges in x direction.
            logical = RotatedPlanar3DPauli(self)

            for x in range(1, 4*Lx+2, 2):
                logical.site('X', (x, 4*Ly + 2 - x, 1))
            logicals.append(logical.to_bsf())

            self._logical_xs = np.array(logicals, dtype=np.uint)

        return self._logical_xs

    @property
    def logical_zs(self) -> np.ndarray:
        """Get the unique logical Z operator."""
        if self._logical_zs.size == 0:
            Lx, L_y, Lz = self.size
            logicals = []

            # Z operators on x edges forming surface normal to x (yz plane).
            logical = RotatedPlanar3DPauli(self)
            for z in range(1, 2*(Lz+1), 2):
                for x in range(1, 4*Lx+2, 2):
                    logical.site('Z', (x, x, z))
            logicals.append(logical.to_bsf())

            self._logical_zs = np.array(logicals, dtype=np.uint)

        return self._logical_zs

    def axis(self, location):
        x, y, z = location

        if location not in self.qubit_index.keys():
            raise ValueError(f'Location {location} does not correspond to a qubit')

        if (z % 2 == 0):
            axis = self.Z_AXIS
        elif (x + y) % 2 == 2:
            axis = self.X_AXIS
        elif (x + y) % 2 == 0:
            axis = self.Y_AXIS

        return axis

    def _create_qubit_indices(self):
        Lx, Ly, Lz = self.size

        coordinates = []

        # Horizontal
        for x in range(1, 4*Lx+2, 2):
            for y in range(1, 4*Ly+2, 2):
                for z in range(1, 2*(Lz+1), 2):
                    coordinates.append((x, y, z))

        # Vertical
        for x in range(2, 4*Lx+1, 2):
            for y in range(0, 4*Ly+3, 2):
                for z in range(2, 2*(Lz+1), 2):
                    if (x + y) % 4 == 2:
                        coordinates.append((x, y, z))

        coord_to_index = {coord: i for i, coord in enumerate(coordinates)}

        return coord_to_index

    def _create_vertex_indices(self):
        Lx, Ly, Lz = self.size

        coordinates = []

        for z in range(1, 2*(Lz+1), 2):
            for x in range(2, 4*Lx+1, 2):
                for y in range(0, 4*Ly+3, 2):
                    if (x + y) % 4 == 2:
                        coordinates.append((x, y, z))

        coord_to_index = {coord: i for i, coord in enumerate(coordinates)}

        return coord_to_index

    def _create_face_indices(self):
        Lx, Ly, Lz = self.size

        coordinates = []

        # Horizontal faces
        for x in range(0, 4*Lx+3, 2):
            for y in range(2, 4*Ly+1, 2):
                for z in range(1, 2*(Lz+1), 2):
                    if (x + y) % 4 == 0:
                        coordinates.append((x, y, z))
        # Vertical faces
        for x in range(1, 4*Lx+3, 2):
            for y in range(1, 4*Ly+2, 2):
                for z in range(2, 2*(Lz+1), 2):
                    coordinates.append((x, y, z))

        coord_to_index = {coord: i for i, coord in enumerate(coordinates)}

        return coord_to_index


if __name__ == "__main__":
    code = RotatedPlanarCode3D(2)

    print("Vertices", code.face_index)
