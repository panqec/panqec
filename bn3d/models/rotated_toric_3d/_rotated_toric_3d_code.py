from typing import Tuple
import numpy as np
from bn3d.models import StabilizerCode
from ._rotated_toric_3d_pauli import RotatedToric3DPauli
from ... import bsparse


class RotatedToric3DCode(StabilizerCode):
    """Rotated Toric Code for good subthreshold scaling."""

    pauli_class = RotatedToric3DPauli

    @property
    def label(self) -> str:
        return 'Rotated Toric 3D {}x{}x{}'.format(*self.size)

    @property
    def dimension(self) -> int:
        return 3

    @property
    def logical_xs(self) -> np.ndarray:
        """Get the unique logical X operator."""

        Lx, Ly, Lz = self.size
        logicals = bsparse.empty_row(2*self.n)

        # Even times even.
        if Lx % 2 == 0 and Ly % 2 == 0:
            # X string operator along y.
            logical = self.pauli_class(self)
            for x, y, z in self.qubit_index:
                if y == 0 and z == 1:
                    logical.site('X', (x, y, z))
            logicals = bsparse.vstack([logicals, logical.to_bsf()])

            # X string operator along x.
            logical = self.pauli_class(self)
            for x, y, z in self.qubit_index:
                if x == 0 and z == 1:
                    logical.site('X', (x, y, z))

            logicals = bsparse.vstack([logicals, logical.to_bsf()])

        # Odd times odd
        elif Lx % 2 == 1 and Ly % 2 == 1:
            logical = self.pauli_class(self)

            for x, y, z in self.qubit_index:
                # X string operator in undeformed code. (OK)
                if z == 1 and x + y == 2*Lx-2:
                    logical.site('X', (x, y, z))

            logicals = bsparse.vstack([logicals, logical.to_bsf()])

        # Odd times even.
        else:
            logical = self.pauli_class(self)

            for x, y, z in self.qubit_index:
                # X string operator in undeformed code. (OK)
                if Lx % 2 == 1:
                    if z == 1 and x == 0:
                        if (x + y) % 4 == 0:
                            logical.site('X', (x, y, z))
                        else:
                            logical.site('X', (x, y, z))
                else:
                    if z == 1 and y == 0:
                        if (x + y) % 4 == 0:
                            logical.site('X', (x, y, z))
                        else:
                            logical.site('X', (x, y, z))

            logicals = bsparse.vstack([logicals, logical.to_bsf()])

        self._logical_xs = logicals

        return self._logical_xs

    @property
    def logical_zs(self) -> np.ndarray:
        """Get the unique logical Z operator."""
        Lx, Ly, Lz = self.size
        logicals = bsparse.empty_row(2*self.n)

        # Even times even.
        if (Lx % 2 == 0) and (Ly % 2 == 0):
            logical = self.pauli_class(self)
            for x, y, z in self.qubit_index:
                if x == 0:
                    logical.site('Z', (x, y, z))
            logicals = bsparse.vstack([logicals, logical.to_bsf()])

            logical = self.pauli_class(self)
            for x, y, z in self.qubit_index:
                if y == 0:
                    logical.site('Z', (x, y, z))

            logicals = bsparse.vstack([logicals, logical.to_bsf()])

        # Odd times odd
        elif (Lx % 2 == 1) and (Ly % 2 == 1):
            logical = self.pauli_class(self)
            for x, y, z in self.qubit_index:
                if x == y:
                    logical.site('Z', (x, y, z))

            logicals = bsparse.vstack([logicals, logical.to_bsf()])

        # Odd times even
        else:
            logical = self.pauli_class(self)
            for x, y, z in self.qubit_index:
                if Lx % 2 == 1:
                    if y == 0:
                        logical.site('Y', (x, y, z))
                elif Ly % 2 == 1:
                    if x == 0:
                        logical.site('Y', (x, y, z))

            logicals = bsparse.vstack([logicals, logical.to_bsf()])

        self._logical_zs = logicals

        return self._logical_zs

    def axis(self, location) -> int:
        x, y, z = location

        if location not in self.qubit_index:
            raise ValueError(f'Location {location} does not correspond to a qubit')

        if z % 2 == 0:
            axis = self.Z_AXIS
        elif (x + y) % 4 == 2:
            axis = self.X_AXIS
        elif (x + y) % 4 == 0:
            axis = self.Y_AXIS

        return axis

    def _create_qubit_indices(self):
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

    def _create_vertex_indices(self):
        Lx, Ly, Lz = self.size

        coordinates = []

        for x in range(1, 2*Lx, 2):
            for y in range(1, 2*Ly, 2):
                for z in range(1, 2*Lz, 2):
                    if (x + y) % 4 == 0:
                        coordinates.append((x, y, z))

        coord_to_index = {coord: i for i, coord in enumerate(coordinates)}

        return coord_to_index

    def _create_face_indices(self):
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
