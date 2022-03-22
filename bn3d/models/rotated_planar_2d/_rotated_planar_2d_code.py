from typing import Tuple
import numpy as np
from bn3d.models import StabilizerCode
from ._rotated_planar_2d_pauli import RotatedPlanar2DPauli
from ... import bsparse


class RotatedPlanar2DCode(StabilizerCode):

    pauli_class = RotatedPlanar2DPauli

    # StabilizerCode interface methods.

    @property
    def dimension(self) -> int:
        return 2

    @property
    def label(self) -> str:
        return 'Toric {}x{}'.format(*self.size)

    @property
    def logical_xs(self) -> np.ndarray:
        """The 2 logical X operators."""

        if self._logical_xs.size == 0:
            Lx, Ly = self.size
            logicals = bsparse.empty_row(2*self.n)

            # X operators along first diagonal
            logical = self.pauli_class(self)
            for x in range(1, 2*Lx+1, 2):
                logical.site('X', (x, 1))
            logicals = bsparse.vstack([logicals, logical.to_bsf()])

            self._logical_xs = logicals

        return self._logical_xs

    @property
    def logical_zs(self) -> np.ndarray:
        """Get the 3 logical Z operators."""
        if self._logical_zs.size == 0:
            Lx, Ly = self.size
            logicals = bsparse.empty_row(2*self.n)

            # Z operators along first diagonal
            logical = self.pauli_class(self)
            for y in range(1, 2*Ly+1, 2):
                logical.site('Z', (1, y))
            logicals = bsparse.vstack([logicals, logical.to_bsf()])

            self._logical_zs = logicals

        return self._logical_zs

    def axis(self, location):
        x, y = location

        if (x + y) % 4 == 2:
            axis = self.X_AXIS
        elif (x + y) % 4 == 0:
            axis = self.Y_AXIS
        else:
            raise ValueError(f'Location {location} does not correspond to a qubit')

        return axis

    def _create_qubit_indices(self):
        coordinates = []
        Lx, Ly = self.size

        # Qubits along e_x
        for x in range(1, 2*Lx+1, 2):
            for y in range(1, 2*Ly+1, 2):
                coordinates.append((x, y))

        coord_to_index = {coord: i for i, coord in enumerate(coordinates)}

        return coord_to_index

    def _create_vertex_indices(self):
        coordinates = []
        Lx, Ly = self.size

        for x in range(2, 2*Lx, 2):
            for y in range(0, 2*Ly+1, 2):
                if (x + y) % 4 == 2:
                    coordinates.append((x, y))

        coord_to_index = {coord: i for i, coord in enumerate(coordinates)}

        return coord_to_index

    def _create_face_indices(self):
        coordinates = []
        Lx, Ly = self.size

        for x in range(0, 2*Lx+1, 2):
            for y in range(2, 2*Ly, 2):
                if (x + y) % 4 == 0:
                    coordinates.append((x, y))

        coord_to_index = {coord: i for i, coord in enumerate(coordinates)}

        return coord_to_index
