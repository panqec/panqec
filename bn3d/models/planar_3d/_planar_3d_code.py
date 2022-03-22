from typing import Tuple
import numpy as np
from ..generic._indexed_sparse_code import IndexedSparseCode
from ._planar_3d_pauli import Planar3DPauli
from ... import bsparse


class Planar3DCode(IndexedSparseCode):

    pauli_class = Planar3DPauli

    # StabilizerCode interface methods.

    @property
    def dimension(self) -> int:
        return 3

    @property
    def label(self) -> str:
        return 'Planar {}x{}x{}'.format(*self.size)

    @property
    def logical_xs(self) -> np.ndarray:
        """The 1 logical X operator."""

        if self._logical_xs.size == 0:
            Lx, Ly, Lz = self.size
            logicals = bsparse.empty_row(2*self.n)

            # X operators along x edges in x direction.
            logical = self.pauli_class(self)
            for x in range(1, 2*Lx+1, 2):
                logical.site('X', (x, 0, 0))
            logicals = bsparse.vstack([logicals, logical.to_bsf()])

            self._logical_xs = logicals

        return self._logical_xs

    @property
    def logical_zs(self) -> np.ndarray:
        """Get the 1 logical Z operator."""
        if self._logical_zs.size == 0:
            Lx, Ly, Lz = self.size
            logicals = bsparse.empty_row(2*self.n)

            # Z operators on x edges forming surface normal to x (yz plane).
            logical = self.pauli_class(self)
            for y in range(0, 2*Ly, 2):
                for z in range(0, 2*Lz, 2):
                    logical.site('Z', (1, y, z))
            logicals = bsparse.vstack([logicals, logical.to_bsf()])

            self._logical_zs = logicals

        return self._logical_zs

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

    def _create_qubit_indices(self):
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

    def _create_vertex_indices(self):
        coordinates = []
        Lx, Ly, Lz = self.size

        for x in range(2, 2*Lx, 2):
            for y in range(0, 2*Ly, 2):
                for z in range(0, 2*Lz, 2):
                    coordinates.append((x, y, z))

        coord_to_index = {coord: i for i, coord in enumerate(coordinates)}

        return coord_to_index

    def _create_face_indices(self):
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
