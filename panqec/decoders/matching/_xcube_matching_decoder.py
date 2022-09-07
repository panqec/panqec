import numpy as np
from typing import Optional, Tuple
from pymatching import Matching
from panqec.decoders import BaseDecoder, MatchingDecoder
from panqec.codes import StabilizerCode, Toric2DCode
from panqec.decoders.bposd.bposd_decoder import BeliefPropagationOSDDecoder
from panqec.error_models import BaseErrorModel


class XCubeMatchingDecoder(BaseDecoder):
    """Matching decoder for 2D Toric Code, based on PyMatching"""

    label = 'XCube Matching'
    allowed_codes = ["XCubeCode"]

    def __init__(self,
                 code: StabilizerCode,
                 error_model: BaseErrorModel,
                 error_rate: float):
        """Constructor for the MatchingDecoder class

        Parameters
        ----------
        code : StabilizerCode
            Code used by the decoder
        error_model: BaseErrorModel
            Error model used by the decoder (to find the weights)
        error_rate: int, optional
            Error rate used by the decoder (to find the weights)
        error_type: str, optional
            Determines which type of errors (X or Z) to decode.
            Can take the values "X", "Z", or None if we want to
            decode all errors
        """
        super().__init__(code, error_model, error_rate)

        self.z_decoder = BeliefPropagationOSDDecoder(self.code,
                                                     self.error_model,
                                                     self.error_rate)

    def decode(self, syndrome: np.ndarray, **kwargs) -> np.ndarray:
        """Get X corrections given code and measured syndrome."""

        # Initialize correction as full bsf.
        correction = np.zeros(2*self.code.n, dtype=np.uint)

        Lx, Ly, Lz = self.code.size

        toric_code = {'x': Toric2DCode(Ly, Lz),
                      'y': Toric2DCode(Lx, Lz),
                      'z': Toric2DCode(Lx, Ly)
                      }

        decoder = {axis: MatchingDecoder(toric_code[axis],
                                         self.error_model,
                                         self.error_rate)
                   for axis in ['x', 'y', 'z']}

        plane_syndrome = {'x': {x: np.zeros(toric_code['x'].n_stabilizers)
                                for x in range(1, 2*Lx, 2)},
                          'y': {y: np.zeros(toric_code['y'].n_stabilizers)
                                for y in range(1, 2*Ly, 2)},
                          'z': {z: np.zeros(toric_code['z'].n_stabilizers)
                                for z in range(1, 2*Lz, 2)}
                          }

        # Remove X stabilizer syndrome and keep it for later
        x_syndrome = self.code.extract_x_syndrome(syndrome)
        syndrome[self.code.x_indices] = 0

        for i_stab in range(len(self.code.stabilizer_index)):
            if syndrome[i_stab]:
                x, y, z = self.code.stabilizer_coordinates[i_stab]

                idx_face_toric = toric_code['x'].stabilizer_index[(y, z)]
                plane_syndrome['x'][x][idx_face_toric] = 1

                idx_face_toric = toric_code['y'].stabilizer_index[(x, z)]
                plane_syndrome['y'][y][idx_face_toric] = 1

                idx_face_toric = toric_code['z'].stabilizer_index[(x, y)]
                plane_syndrome['z'][z][idx_face_toric] = 1

        xcube_matching_z = []
        xcube_matching_xy = []
        for axis in ['x', 'y', 'z']:
            for plane_id in plane_syndrome[axis].keys():
                current_syndrome = plane_syndrome[axis][plane_id]

                toric_matching = decoder[axis].decode(current_syndrome)

                for i in range(toric_code[axis].n):
                    if toric_matching[toric_code[axis].n+i]:
                        x, y = toric_code[axis].qubit_coordinates[i]
                        if axis == 'z':
                            xcube_matching_z.append((x, y, plane_id))
                        elif axis == 'x':
                            xcube_matching_xy.append((plane_id, x, y))
                        elif axis == 'y':
                            xcube_matching_xy.append((x, plane_id, y))

        if Lz % 2 == 0:
            z_proj = Lz - 1
        else:
            z_proj = Lz

        for (x, y, z) in xcube_matching_z:
            if z < z_proj:
                for z_op in range(z + 1, z_proj + 1, 2):
                    idx = self.code.qubit_index[(x, y, z_op)]
                    correction[idx] += 1
            elif z > z_proj:
                for z_op in range(z_proj+1, z + 1, 2):
                    idx = self.code.qubit_index[(x, y, z_op % (2*Lz))]
                    correction[idx] += 1

        # - Find the loop
        toric_loop_dict = {}
        for (x, y, z) in xcube_matching_xy:
            if (x, y) in toric_loop_dict.keys():
                toric_loop_dict[(x, y)] += 1
            else:
                toric_loop_dict[(x, y)] = 1

        toric_loop = [coord for coord in toric_loop_dict.keys()
                      if toric_loop_dict[coord] % 2 == 1]

        # print(toric_loop)

        # - Distinguish the two sides of the loops
        state = {}
        for x in range(0, 2*Lx, 2):
            current_state = 0
            for y in range(0, 2*Ly, 2):
                if y == 0 and (x - 1, y) in toric_loop:
                    current_state = 1 - state[(x - 2, y)]
                if (x, y - 1) in toric_loop:
                    current_state = 1 - current_state

                # print((x, y), current_state)
                state[(x, y)] = current_state
                idx = self.code.qubit_index[(x, y, z_proj)]
                correction[idx] = current_state

        # Decode z part
        syndrome[self.code.z_indices] = 0
        syndrome[self.code.x_indices] = x_syndrome

        z_correction = self.z_decoder.decode(syndrome).astype('uint8')

        correction += z_correction

        return correction % 2

    def get_weights(self, eps=1e-10) -> Tuple[np.ndarray, np.ndarray]:
        """Get MWPM weights for deformed Pauli noise."""

        pi, px, py, pz = self.error_model.probability_distribution(
            self.code, self.error_rate
        )

        total_p_x = px + py
        total_p_z = pz + py

        weights_x = -np.log(
            (total_p_x + eps) / (1 - total_p_x + eps)
        )
        weights_z = -np.log(
            (total_p_z + eps) / (1 - total_p_z + eps)
        )

        return weights_x, weights_z
