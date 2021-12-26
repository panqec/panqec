import functools
from typing import Tuple
import numpy as np
from qecsim.model import StabilizerCode
from ..noise import PauliErrorModel


class DeformedRhombicErrorModel(PauliErrorModel):
    """Pauli error model with qubits deformed."""

    # _undeformed_model: PauliErrorModel

    def __init__(self, r_x, r_y, r_z):
        super().__init__(r_x, r_y, r_z)

    @property
    def label(self) -> str:
        return 'Deformed Rhombic Pauli X{:.4f}Y{:.4f}Z{:.4f}'.format(
            *self.direction
        )

    @functools.lru_cache()
    def probability_distribution(
        self, code: StabilizerCode, probability: float
    ) -> Tuple:
        r_x, r_y, r_z = self.direction
        is_deformed = self._get_deformation_indices(code)

        p_i = np.array([1 - probability for _ in range(code.n_k_d[0])])
        p_x = probability * np.array([
            r_z if is_deformed[i] else r_x for i in range(code.n_k_d[0])
        ])
        p_y = probability * np.array([r_y for _ in range(code.n_k_d[0])])
        p_z = probability * np.array([
            r_x if is_deformed[i] else r_z for i in range(code.n_k_d[0])
        ])

        return p_i, p_x, p_y, p_z

    def _get_deformation_indices(self, code: StabilizerCode):
        is_deformed = [False for _ in range(code.n_k_d[0])]

        deformed_axis = {'RhombicCode': code.Z_AXIS}

        if code.id not in deformed_axis.keys():
            raise NotImplementedError(f"Code {code.id} has no rhombic deformation implemented")

        for (x, y, z) in code.qubit_index.keys():
            if code.axis((x, y, z)) == deformed_axis[code.id] and (x + y) % 4 == 2:
                index = code.qubit_index[(x, y, z)]
                is_deformed[index] = True

        return is_deformed
