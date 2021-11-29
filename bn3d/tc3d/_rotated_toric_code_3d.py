from typing import Tuple, Optional, Dict
import numpy as np
from ._rotated_toric_3d_pauli import RotatedToric3DPauli
from ._indexed_code import IndexedCode
from bn3d.bpauli import bcommute


class RotatedToricCode3D(IndexedCode):

    _size: Tuple[int, int, int]
    _qubit_index: Dict[Tuple[int, int, int], int]
    _vertex_index: Dict[Tuple[int, int, int], int]
    _face_index: Dict[Tuple[int, int, int], int]
    _stabilizers = np.array([])
    _Hx = np.array([])
    _Hz = np.array([])
    _logical_xs = np.array([])
    _logical_zs = np.array([])
    pauli_class = RotatedToric3DPauli

    # StabilizerCode interface methods.
    @property
    def n_k_d(self) -> Tuple[int, int, int]:
        Lx, Ly, Lz = self.size
        n_horizontals = 2*Ly*(2*Lx + 1)*Lz
        n_verticals = (2*Lx*Ly)*Lz
        return (n_horizontals + n_verticals, 3, -1)

    @property
    def label(self) -> str:
        return 'Rotated Toric {}x{}x{}'.format(*self.size)

    @property
    def logical_xs(self) -> np.ndarray:
        """Get the unique logical X operator."""

        if self._logical_xs.size == 0:
            Lx, Ly, Lz = self.size
            logicals = []

            logical = RotatedToric3DPauli(self)
            for y in range(1, 4*Ly, 2):
                logical.site('X', (1, y, 1))
            logicals.append(logical.to_bsf())

            logical = RotatedToric3DPauli(self)
            for x in range(1, 4*Lx, 2):
                logical.site('X', (x, 1, 1))
            logicals.append(logical.to_bsf())

            logical = RotatedToric3DPauli(self)
            for z in range(0, 2*Lz, 2):
                logical.site('X', (2, 0, z))
            logicals.append(logical.to_bsf())

            self._logical_xs = np.array(logicals, dtype=np.uint)

        return self._logical_xs

    @property
    def logical_zs(self) -> np.ndarray:
        """Get the unique logical Z operator."""
        if self._logical_zs.size == 0:
            Lx, Ly, Lz = self.size
            logicals = []

            # Z operators on x edges forming surface normal to x (yz plane).
            logical = RotatedToric3DPauli(self)
            for (x, y, z) in self.qubit_index.keys():
                if z % 2 == 1 and (x + y) % 4 == 2:
                    logical.site('Z', (x, y, z))
            logicals.append(logical.to_bsf())

            logical = RotatedToric3DPauli(self)
            for (x, y, z) in self.qubit_index.keys():
                if z % 2 == 1 and (x + y) % 4 == 0:
                    logical.site('Z', (x, y, z))
            logicals.append(logical.to_bsf())

            logical = RotatedToric3DPauli(self)
            for (x, y, z) in self.qubit_index.keys():
                if z == 0:
                    logical.site('Z', (x, y, z))
            logicals.append(logical.to_bsf())

            self._logical_zs = np.array(logicals, dtype=np.uint)

        return self._logical_zs

    def _create_qubit_indices(self):
        Lx, Ly, Lz = self.size

        coordinates = []

        # Horizontal
        for x in range(1, 4*Lx+2, 2):
            for y in range(1, 4*Ly, 2):
                for z in range(1, 2*Lz, 2):
                    coordinates.append((x, y, z))

        # Vertical
        for x in range(2, 4*Lx+2, 2):
            for y in range(0, 4*Ly, 2):
                for z in range(0, 2*Lz, 2):
                    if (x + y) % 4 == 2:
                        coordinates.append((x, y, z))

        coord_to_index = {coord: i for i, coord in enumerate(coordinates)}

        return coord_to_index

    def _create_vertex_indices(self):
        Lx, Ly, Lz = self.size

        coordinates = []

        for z in range(1, 2*Lz, 2):
            for x in range(2, 4*Lx+1, 2):
                for y in range(0, 4*Ly, 2):
                    if (x + y) % 4 == 2:
                        coordinates.append((x, y, z))

        coord_to_index = {coord: i for i, coord in enumerate(coordinates)}

        return coord_to_index

    def _create_face_indices(self):
        Lx, Ly, Lz = self.size

        coordinates = []

        # Horizontal faces
        for x in range(0, 4*Lx+3, 2):
            for y in range(0, 4*Ly-1, 2):
                for z in range(1, 2*Lz, 2):
                    if (x + y) % 4 == 0:
                        coordinates.append((x, y, z))
        # Vertical faces
        for x in range(1, 4*Lx+2, 2):
            for y in range(1, 4*Ly, 2):
                for z in range(0, 2*Lz, 2):
                    coordinates.append((x, y, z))

        coord_to_index = {coord: i for i, coord in enumerate(coordinates)}

        return coord_to_index


if __name__ == "__main__":
    code = RotatedToricCode3D(2)

    print("Vertices", code.face_index)
