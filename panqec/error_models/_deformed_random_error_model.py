from typing import Tuple
import numpy as np
from panqec.codes import StabilizerCode
from . import PauliErrorModel


class DeformedRandomErrorModel(PauliErrorModel):
    """Pauli error model with qubits randomly deformed."""

    ID = 0
    XZ = 1
    YZ = 2

    def __init__(self, r_x, r_y, r_z, p_xz, p_yz):
        super().__init__(r_x, r_y, r_z)

        self.p_xz = np.around(p_xz, 15)
        self.p_yz = np.around(p_yz, 15)
        self.p_id = np.around(1 - p_xz - p_yz, 15)

    @property
    def label(self) -> str:
        return 'Deformed Random Pauli X{:.4f}Y{:.4f}Z{:.4f}'.format(
            *self.direction
        )

    def probability_distribution(
        self, code: StabilizerCode, probability: float
    ) -> Tuple:
        r_x, r_y, r_z = self.direction
        deformations = self._get_deformations(code)

        p_i = np.array([1 - probability for _ in range(code.n)])
        p_x = probability * r_x * np.ones(code.n)
        p_y = probability * r_y * np.ones(code.n)
        p_z = probability * r_z * np.ones(code.n)

        xz_idx = (deformations == self.XZ)
        yz_idx = (deformations == self.YZ)

        p_x[xz_idx], p_z[xz_idx] = p_z[xz_idx], p_x[xz_idx]
        p_y[yz_idx], p_z[yz_idx] = p_z[yz_idx], p_y[yz_idx]

        return p_i, p_x, p_y, p_z

    def _get_deformations(self, code: StabilizerCode):
        deformations = np.random.choice([self.ID, self.XZ, self.YZ],
                                        p=[self.p_id, self.p_xz, self.p_yz],
                                        size=code.n)
        return deformations
