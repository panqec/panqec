import functools
from typing import Tuple
import numpy as np
from qecsim.model import StabilizerCode
from ..noise import PauliErrorModel


class DeformedXZZXErrorModel(PauliErrorModel):
    """Pauli error model with qubits deformed."""

    # _undeformed_model: PauliErrorModel

    def __init__(self, r_x, r_y, r_z):
        super().__init__(r_x, r_y, r_z)

    @property
    def label(self) -> str:
        return 'Deformed XZZX Pauli X{:.4f}Y{:.4f}Z{:.4f}'.format(
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

        deformed_axis = {'ToricCode3D': code.Z_AXIS,
                         'PlanarCode3D': code.Z_AXIS,
                         'XCubeCode': code.Z_AXIS,
                         'RotatedToricCode3D': code.Z_AXIS,
                         'RotatedPlanarCode3D': code.Z_AXIS,
                         'RhombicCode': code.Z_AXIS,
                         'ToricCode2D': code.X_AXIS,
                         'Planar2DCode': code.X_AXIS,
                         'RotatedPlanar2DCode': code.X_AXIS}

        if code.id not in deformed_axis.keys():
            raise NotImplementedError(f"Code {code.id} has no XZZX deformation implemented")

        # if "Layered Rotated" in code.label:
        #     for (x, y, z), index in code.qubit_index.items():
        #         if z % 2 == 1 and (x + y) % 4 == 2:
        #             is_deformed[index] = True
        # elif "Rotated" in code.label:
        #     for (x, y, z), index in code.qubit_index.items():
        #         if z % 2 == 0:
        #             is_deformed[index] = True
        # else:
        for location, index in code.qubit_index.items():
            if code.axis(location) == deformed_axis[code.id]:
                is_deformed[index] = True

        return is_deformed
