from typing import Tuple
import numpy as np
from ._indexed_code import IndexedCode
from ._layered_toric_pauli import LayeredToricPauli


class LayeredRotatedToricCode(IndexedCode):
    """Layered Rotated Code for good subthreshold scaling."""

    pauli_class = LayeredToricPauli

    @property
    def label(self) -> str:
        return 'Layered Rotated Toric {}x{}x{}'.format(*self.size)

    @property
    def n_k_d(self) -> Tuple[int, int, int]:
        L_x, L_y, L_z = self.size
        x_odd = L_x % 2 == 1
        y_odd = L_y % 2 == 1
        k = 2
        if x_odd or y_odd:
            k = 1
        return (len(self.qubit_index), k, min(L_x, L_y))

    @property
    def logical_xs(self) -> np.ndarray:
        """Get the unique logical X operator."""

        if self._logical_xs.size == 0:
            L_x, L_y, L_z = self.size
            logicals = []

            # Even times even.
            if L_x % 2 == 0 and L_y % 2 == 0:

                # X string operator along y.
                logical = self.pauli_class(self)
                for x, y, z in self.qubit_index:
                    if y == 1 and z == 1:
                        logical.site('X', (x, y, z))
                logicals.append(logical.to_bsf())

                # X string operator along x.
                logical = self.pauli_class(self)
                for x, y, z in self.qubit_index:
                    if x == 1 and z == 1:
                        logical.site('X', (x, y, z))
                logicals.append(logical.to_bsf())

            # Odd times even.
            else:
                logical = self.pauli_class(self)
                for x, y, z in self.qubit_index:
                    # Z on every layer in deformed code. (FAIL, in stabilizer)
                    """
                    if z % 2 == 1:
                        if (x + y) % 4 == 2:
                            logical.site('X', (x, y, z))
                        else:
                            logical.site('Z', (x, y, z))
                    """

                    # X string operator in undeformed code. (OK)
                    if L_x % 2 == 1:
                        if z == 1 and x == 1:
                            if (x + y) % 4 == 2:
                                logical.site('X', (x, y, z))
                            else:
                                logical.site('X', (x, y, z))
                    else:
                        if z == 1 and y == 1:
                            if (x + y) % 4 == 2:
                                logical.site('X', (x, y, z))
                            else:
                                logical.site('X', (x, y, z))

                    # Z everywhere in deformed (FAIL, actually in stabilizer)
                    """
                    if z % 2 == 1:
                        if (x + y) % 4 == 2:
                            logical.site('X', (x, y, z))
                        else:
                            logical.site('Z', (x, y, z))
                    else:
                        logical.site('Z', (x, y, z))
                    """
                logicals.append(logical.to_bsf())

            self._logical_xs = np.array(logicals, dtype=np.uint)

        return self._logical_xs

    @property
    def logical_zs(self) -> np.ndarray:
        """Get the unique logical Z operator."""
        if self._logical_zs.size == 0:
            L_x, L_y, L_z = self.size
            logicals = []

            # Even times even.
            if L_x % 2 == 0 and L_y % 2 == 0:

                logical = self.pauli_class(self)
                for x, y, z in self.qubit_index:
                    if x == 1:
                        logical.site('Z', (x, y, z))

                logicals.append(logical.to_bsf())

                logical = self.pauli_class(self)
                for x, y, z in self.qubit_index:
                    if y == 1:
                        logical.site('Z', (x, y, z))

                logicals.append(logical.to_bsf())

            # Odd times even.
            else:
                logical = self.pauli_class(self)
                for x, y, z in self.qubit_index:
                    if L_x % 2 == 1:
                        if y == 1:
                            logical.site('Y', (x, y, z))
                    elif L_y % 2 == 1:
                        if x == 1:
                            logical.site('Y', (x, y, z))
                logicals.append(logical.to_bsf())

            self._logical_zs = np.array(logicals, dtype=np.uint)

        return self._logical_zs

    def _create_qubit_indices(self):
        L_x, L_y, L_z = self.size

        coordinates = []

        # Horizontal
        for x in range(1, 2*L_x, 2):
            for y in range(1, 2*L_y, 2):
                for z in range(1, 2*L_z + 2, 2):
                    coordinates.append((x, y, z))

        # Vertical
        for x in range(2, 2*L_x + 1, 2):
            for y in range(2, 2*L_y + 1, 2):
                for z in range(2, 2*L_z + 2, 2):
                    if (x + y) % 4 == 2:
                        coordinates.append((x, y, z))

        coord_to_index = {coord: i for i, coord in enumerate(coordinates)}

        return coord_to_index

    def _create_vertex_indices(self):
        L_x, L_y, L_z = self.size

        coordinates = []

        for x in range(2, 2*L_x + 1, 2):
            for y in range(2, 2*L_y + 1, 2):
                for z in range(1, 2*L_z + 2, 2):
                    if (x + y) % 4 == 2:
                        coordinates.append((x, y, z))

        coord_to_index = {coord: i for i, coord in enumerate(coordinates)}

        return coord_to_index

    def _create_face_indices(self):
        L_x, L_y, L_z = self.size

        coordinates = []

        # Horizontal faces
        for x in range(2, 2*L_x + 1, 2):
            for y in range(2, 2*L_y + 1, 2):
                for z in range(1, 2*L_z + 2, 2):
                    if (x + y) % 4 == 0:
                        coordinates.append((x, y, z))

        # Vertical faces
        for x in range(1, 2*L_x, 2):
            for y in range(1, 2*L_y, 2):
                for z in range(2, 2*L_z + 2, 2):
                    coordinates.append((x, y, z))

        coord_to_index = {coord: i for i, coord in enumerate(coordinates)}

        return coord_to_index
