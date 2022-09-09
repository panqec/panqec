import numpy as np
from typing import Optional, Tuple
from pymatching import Matching
from panqec.decoders import BaseDecoder, MatchingDecoder
from panqec.codes import StabilizerCode, Toric2DCode
from panqec.decoders.bposd.bposd_decoder import BeliefPropagationOSDDecoder
from panqec.error_models import PauliErrorModel


def modular_average(numbers, weights, n):
    if isinstance(numbers, list):
        numbers = np.array(numbers)
    if isinstance(weights, list):
        weights = np.array(weights)

    weights = weights.reshape(-1, 1)

    circle_points = np.vstack([np.cos(2 * np.pi * numbers / n),
                               np.sin(2 * np.pi * numbers / n)]).T

    mass_center = np.mean(weights * circle_points, axis=0)
    if np.all(np.isclose(mass_center, 0)):
        avg = np.dot(numbers, weights)
    else:
        circle_avg = mass_center / np.linalg.norm(mass_center)
        avg = np.arccos(circle_avg[0]) * n / (2 * np.pi)

        if circle_avg[1] < 0:
            avg *= -1

    if avg < 0:
        avg += n

    return avg


if __name__ == "__main__":
    avg = modular_average([1, 3, 5, 7], [0.5, 0.0, 0.5, 0.0], 8)

    print(avg)


class XCubeMatchingDecoder(BaseDecoder):
    """Matching decoder for 2D Toric Code, based on PyMatching"""

    label = 'XCube Matching'
    allowed_codes = ["XCubeCode"]

    def __init__(self,
                 code: StabilizerCode,
                 error_model: PauliErrorModel,
                 error_rate: float):
        """Constructor for the MatchingDecoder class

        Parameters
        ----------
        code : StabilizerCode
            Code used by the decoder
        error_model: PauliErrorModel
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

        rx, ry, rz = self.error_model.direction

        # Weight the 2D toric code matching decoders
        # Only works for Z biased noise and z-axis deformation
        weights_X, _ = self.error_model.get_weights(self.code,
                                                    self.error_rate)

        wz = weights_X[self.code.qubit_index[(0, 0, 1)]]
        wxy = weights_X[self.code.qubit_index[(1, 0, 0)]]

        weights = {'x': [wxy if toric_code['x'].qubit_axis(loc) == 'x' else wz
                         for loc in toric_code['x'].qubit_coordinates],
                   'y': [wxy if toric_code['y'].qubit_axis(loc) == 'x' else wz
                         for loc in toric_code['y'].qubit_coordinates],
                   'z': [wxy for _ in toric_code['z'].qubit_coordinates]}

        decoder = {axis: MatchingDecoder(toric_code[axis],
                                         self.error_model,
                                         self.error_rate,
                                         weights=(weights[axis],
                                                  weights[axis]))
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

        # Decode all the 2D toric codes
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

        # Find the optimal projection plane
        n_excitations = np.sum(syndrome)
        n_excitations_plane = {z: np.sum(plane_syndrome['z'][z])
                               for z in plane_syndrome['z'].keys()}

        average_z = 0

        if n_excitations != 0:
            weights = [n_excitations_plane[z] / n_excitations
                       for z in n_excitations_plane.keys()]
            plane_z = list(n_excitations_plane.keys())

            average_z = modular_average(plane_z, weights, 2*Lz)

        z_proj = int(2 * np.floor(average_z/2) + 1)
        # print("Z proj", z_proj)

        # Perform the projection
        for (x, y, z) in xcube_matching_z:
            # print("Matching", (x, y, z))
            if 0 < z - z_proj <= Lz or 2*Lz + z - z_proj <= Lz:
                # print("test 1")
                if z_proj < z:
                    z_stop = z + 1
                else:
                    z_stop = 2*Lz + z + 1

                for z_op in range(z_proj+1, z_stop, 2):
                    idx = self.code.qubit_index[(x, y, z_op % (2*Lz))]
                    correction[idx] += 1
            elif z != z_proj:
                # print("test 2")
                if z < z_proj:
                    z_stop = z_proj + 1
                else:
                    z_stop = 2*Lz + z_proj + 1

                for z_op in range(z + 1, z_stop, 2):
                    idx = self.code.qubit_index[(x, y, z_op % (2*Lz))]
                    correction[idx] += 1

        # Find the loops
        toric_loop_dict = {}
        for (x, y, z) in xcube_matching_xy:
            # print("xy matching", (x, y, z))
            if (x, y) in toric_loop_dict.keys():
                toric_loop_dict[(x, y)] += 1
            else:
                toric_loop_dict[(x, y)] = 1

        toric_loop = [coord for coord in toric_loop_dict.keys()
                      if toric_loop_dict[coord] % 2 == 1]

        # print("Toric loop", toric_loop)

        # Distinguish the two sides of the loops
        state = {}
        for x in range(0, 2*Lx, 2):
            current_state = 0
            for y in range(0, 2*Ly, 2):
                if y == 0:
                    if (x - 1, y) in toric_loop:
                        current_state = 1 - state[(x - 2, y)]
                    elif x >= 2:
                        current_state = state[(x - 2, y)]
                if (x, y - 1) in toric_loop:
                    current_state = 1 - current_state

                # print((x, y), current_state)
                state[(x, y)] = current_state
                idx = self.code.qubit_index[(x, y, z_proj)]

        # Correct the minority vote
        count = np.unique(list(state.values()), return_counts=True)

        if len(count[0]) == 1:
            minority_state = 1 - count[0]
        else:
            minority_state = count[0][np.argmin(count[1])]

        # print(minority_state)

        for x in range(0, 2*Lx, 2):
            for y in range(0, 2*Ly, 2):
                if state[(x, y)] == minority_state:
                    idx = self.code.qubit_index[(x, y, z_proj)]
                    correction[idx] = 1

        # Decode z part
        syndrome[self.code.z_indices] = 0
        syndrome[self.code.x_indices] = x_syndrome

        z_correction = self.z_decoder.decode(syndrome).astype('uint8')

        correction += z_correction

        return correction % 2
