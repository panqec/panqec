import functools
from typing import Tuple
import numpy as np
from panqec.codes import StabilizerCode
from . import PauliErrorModel


class DeformedColorErrorModel(PauliErrorModel):
    """Pauli error model with qubits deformed."""

    # _undeformed_model: PauliErrorModel

    def __init__(self, r_x, r_y, r_z):
        super().__init__(r_x, r_y, r_z)

    @property
    def label(self) -> str:
        return 'Deformed Color Pauli X{:.4f}Y{:.4f}Z{:.4f}'.format(
            *self.direction
        )

    @functools.lru_cache()
    def probability_distribution(
        self, code: StabilizerCode, error_rate: float
    ) -> Tuple:
        r_x, r_y, r_z = self.direction
        is_deformed = self.get_deformation_indices(code)

        p_i = np.array([1 - error_rate for _ in range(code.n)])
        p_x = error_rate * np.array([
            r_z if is_deformed[i] else r_x for i in range(code.n)
        ])
        p_y = error_rate * np.array([r_y for _ in range(code.n)])
        p_z = error_rate * np.array([
            r_x if is_deformed[i] else r_z for i in range(code.n)
        ])

        return p_i, p_x, p_y, p_z

    def get_deformation_indices(self, code: StabilizerCode):
        is_deformed = [False for _ in range(code.n)]

        if code.id == 'Color3DCode':
            Lx, Ly, Lz = code.size

            for x, y, z in code.type_index('face-square').keys():
                # Squares on green and blue cells, orthogonal to x axis
                if x % 4 == 2 and y % 4 == 0 and z % 4 == 0:
                    for qubit_loc in code.get_stabilizer((x, y, z)):
                        index = code.qubit_index[qubit_loc]
                        is_deformed[index] = True

        elif code.id == 'Color666Code':
            Lx, Ly = code.size

            # Squares on green and blue cells, orthogonal to x axis
            for x in range(2, 4*Lx, 6):
                for y in range(2, min(2*x + 1, 4*Lx - 2*x - 3), 4):
                    if x - 1 >= 0:
                        index = code.qubit_index[(x-1, y)]
                        is_deformed[index] = True
                    if x + 1 < 4*Lx:
                        index = code.qubit_index[(x+1, y)]
                        is_deformed[index] = True

        else:
            raise NotImplementedError(f"Code {code.id} has no color code\
                                      deformation implemented")

        return is_deformed
