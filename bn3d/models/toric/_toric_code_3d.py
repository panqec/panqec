from typing import Tuple
import numpy as np
from ..generic._indexed_code import IndexedCode
from ._toric_3d_pauli import Toric3DPauli


class ToricCode3D(IndexedCode):

    pauli_class = Toric3DPauli

    # StabilizerCode interface methods.

    @property
    def n_k_d(self) -> Tuple[int, int, int]:
        return (3 * np.product(self.size), 3, min(self.size))

    @property
    def label(self) -> str:
        return 'Toric {}x{}x{}'.format(*self.size)

    @property
    def logical_xs(self) -> np.ndarray:
        """The 3 logical X operators."""

        if self._logical_xs.size == 0:
            Lx, Ly, Lz = self.size
            logicals = []

            # X operators along x edges in x direction.
            logical = Toric3DPauli(self)
            for x in range(1, 2*Lx, 2):
                logical.site('X', (x, 0, 0))
            logicals.append(logical.to_bsf())

            # X operators along y edges in y direction.
            logical = Toric3DPauli(self)
            for y in range(1, 2*Ly, 2):
                logical.site('X', (0, y, 0))
            logicals.append(logical.to_bsf())

            # X operators along z edges in z direction
            logical = Toric3DPauli(self)
            for z in range(1, 2*Lz, 2):
                logical.site('X', (0, 0, z))
            logicals.append(logical.to_bsf())

            self._logical_xs = np.array(logicals, dtype=np.uint)

        return self._logical_xs

    @property
    def logical_zs(self) -> np.ndarray:
        """Get the 3 logical Z operators."""
        if self._logical_zs.size == 0:
            Lx, Ly, Lz = self.size
            logicals = []

            # Z operators on x edges forming surface normal to x (yz plane).
            logical = Toric3DPauli(self)
            for y in range(0, 2*Ly, 2):
                for z in range(0, 2*Lz, 2):
                    logical.site('Z', (1, y, z))
            logicals.append(logical.to_bsf())

            # Z operators on y edges forming surface normal to y (zx plane).
            logical = Toric3DPauli(self)
            for z in range(0, 2*Lz, 2):
                for x in range(0, 2*Lx, 2):
                    logical.site('Z', (x, 1, z))
            logicals.append(logical.to_bsf())

            # Z operators on z edges forming surface normal to z (xy plane).
            logical = Toric3DPauli(self)
            for x in range(0, 2*Lx, 2):
                for y in range(0, 2*Ly, 2):
                    logical.site('Z', (x, y, 1))
            logicals.append(logical.to_bsf())

            self._logical_zs = np.array(logicals, dtype=np.uint)

        return self._logical_zs

    def _create_qubit_indices(self):
        coordinates = []
        Lx, Ly, Lz = self.size

        # Qubits along e_x
        for x in range(1, 2*Lx, 2):
            for y in range(0, 2*Ly, 2):
                for z in range(0, 2*Lz, 2):
                    coordinates.append((x, y, z))

        # Qubits along e_y
        for x in range(0, 2*Lx, 2):
            for y in range(1, 2*Ly, 2):
                for z in range(0, 2*Lz, 2):
                    coordinates.append((x, y, z))

        # Qubits along e_z
        for x in range(0, 2*Lx, 2):
            for y in range(0, 2*Ly, 2):
                for z in range(1, 2*Lz, 2):
                    coordinates.append((x, y, z))

        coord_to_index = {coord: i for i, coord in enumerate(coordinates)}

        return coord_to_index

    def _create_vertex_indices(self):
        coordinates = []
        Lx, Ly, Lz = self.size

        for x in range(0, 2*Lx):
            for y in range(0, 2*Ly):
                for z in range(0, 2*Lz, 2):
                    if (x % 2 == 0) and (y % 2 == 0):
                        coordinates.append((x, y, z))

        coord_to_index = {coord: i for i, coord in enumerate(coordinates)}

        return coord_to_index

    def _create_face_indices(self):
        coordinates = []
        Lx, Ly, Lz = self.size

        # Face in xy plane
        for x in range(1, 2*Lx, 2):
            for y in range(1, 2*Ly, 2):
                for z in range(0, 2*Lz, 2):
                    coordinates.append((x, y, z))

        # Face in yz plane
        for x in range(0, 2*Lx, 2):
            for y in range(1, 2*Ly, 2):
                for z in range(1, 2*Lz, 2):
                    coordinates.append((x, y, z))

        # Face in xz plane
        for x in range(1, 2*Lx, 2):
            for y in range(0, 2*Ly, 2):
                for z in range(1, 2*Lz, 2):
                    coordinates.append((x, y, z))

        coord_to_index = {coord: i for i, coord in enumerate(coordinates)}

        return coord_to_index
