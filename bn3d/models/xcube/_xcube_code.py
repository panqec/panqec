import itertools
from typing import Tuple
import numpy as np
from ..generic._indexed_sparse_code import IndexedSparseCode
from ._xcube_pauli import XCubePauli
from ... import bsparse


class XCubeCode(IndexedSparseCode):

    pauli_class = XCubePauli

    # StabilizerCode interface methods.

    @property
    def dimension(self) -> int:
        return 3

    @property
    def label(self) -> str:
        return 'XCube {}x{}x{}'.format(*self.size)

    @property
    def logical_xs(self) -> np.ndarray:
        """The 3 logical X operators."""

        if self._logical_xs.size == 0:
            Lx, Ly, Lz = self.size
            logicals = bsparse.empty_row(2*self.n)

            # String of parallel X operators along the x direction
            logical = self.pauli_class(self)
            for x in range(0, 2*Lx, 2):
                logical.site('X', (x, 1, 0))
            logicals = bsparse.vstack([logicals, logical.to_bsf()])

            # String of parallel X operators normal to the y direction
            logical = self.pauli_class(self)
            for y in range(0, 2*Ly, 2):
                logical.site('X', (1, y, 0))
            logicals = bsparse.vstack([logicals, logical.to_bsf()])

            # String of parallel X operators normal to the z direction
            logical = self.pauli_class(self)
            for z in range(0, 2*Lz, 2):
                logical.site('X', (0, 1, z))
            logicals = bsparse.vstack([logicals, logical.to_bsf()])

            self._logical_xs = bsparse.from_array(logicals)

        return self._logical_xs

    @property
    def logical_zs(self) -> np.ndarray:
        """Get the 3 logical Z operators."""
        if self._logical_zs.size == 0:
            Lx, Ly, Lz = self.size
            logicals = bsparse.empty_row(2*self.n)

            # Line of parallel Z operators along the x direction
            logical = self.pauli_class(self)
            for x in range(1, 2*Lx, 2):
                logical.site('Z', (x, 0, 0))
            logicals = bsparse.vstack([logicals, logical.to_bsf()])

            # Line of parallel Z operators along the y direction
            logical = self.pauli_class(self)
            for x in range(1, 2*Lx, 2):
                logical.site('Z', (0, x, 0))
            logicals = bsparse.vstack([logicals, logical.to_bsf()])

            # Line of parallel Z operators along the z direction
            logical = self.pauli_class(self)
            for z in range(1, 2*Lz, 2):
                logical.site('Z', (0, 0, z))
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
        """ Vertex = cube stabilizer"""
        Lx, Ly, Lz = self.size

        ranges = [range(1, 2*Lx, 2), range(1, 2*Ly, 2), range(1, 2*Lz, 2)]
        coordinates = []
        for x, y, z in itertools.product(*ranges):
            coordinates.append((x, y, z))

        coord_to_index = {coord: i for i, coord in enumerate(coordinates)}

        return coord_to_index

    def _create_face_indices(self):
        """ Face stabilizer (three at each vertex)"""

        coordinates = []
        Lx, Ly, Lz = self.size

        for x in range(0, 2*Lx, 2):
            for y in range(0, 2*Ly, 2):
                for z in range(0, 2*Lz, 2):
                    for axis in range(3):
                        coordinates.append((axis, x, y, z))

        coord_to_index = {coord: i for i, coord in enumerate(coordinates)}

        return coord_to_index
