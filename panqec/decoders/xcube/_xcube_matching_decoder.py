import numpy as np
from typing import List, Dict
from panqec.decoders import BaseDecoder, MatchingDecoder
from panqec.codes import StabilizerCode, Toric2DCode
from panqec.decoders import BeliefPropagationOSDDecoder
from panqec.error_models import PauliErrorModel


def get_matched_pairs(H, correction, syndrome):
    pairs = []
    seen_syndromes = set([])
    syndrome_indices = np.nonzero(syndrome)[0]

    for s in syndrome_indices:
        if s not in seen_syndromes:
            # print("s", s)
            seen_syndromes.add(s)

            s_prime = s
            continue_search = True
            prev_qubit = -1

            while continue_search:
                found_new_qubit = False
                for q in H[s_prime].nonzero()[1]:
                    # print("q", q)
                    if correction[q] and q != prev_qubit:
                        # print("Found new qubit")
                        found_new_qubit = True
                        prev_qubit = q
                        for i in H[:, q].nonzero()[0]:
                            if i != s_prime:
                                s_prime = i
                                break

                        break

                if not found_new_qubit:
                    continue_search = False
                    pairs.append((s, s_prime))
                    seen_syndromes.add(s_prime)

    return pairs


def find_connected_components(neighbors):
    connected_components = []
    seen = set()

    def component(node):
        nodes_in_component = set([node])
        nodes = set([node])
        while nodes:
            node = nodes.pop()
            seen.add(node)
            nodes.update(neighbors[node] - seen)
            nodes_in_component.add(node)

        return list(nodes_in_component)

    for node in neighbors:
        if node not in seen:
            connected_components.append(component(node))

    return connected_components


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


def decode_plane(loops, size):
    Lx, Ly = size

    # Distinguish the two sides of the loops
    state = {}
    for x in range(0, 2*Lx, 2):
        current_state = 0
        for y in range(0, 2*Ly, 2):
            if y == 0:
                if (x - 1, y) in loops:
                    current_state = 1 - state[(x - 2, y)]
                elif x >= 2:
                    current_state = state[(x - 2, y)]
            if (x, y - 1) in loops:
                current_state = 1 - current_state

            # print((x, y), current_state)
            state[(x, y)] = current_state

    # Correct the minority vote
    count = np.unique(list(state.values()), return_counts=True)

    if len(count[0]) == 1:
        minority_state = 1 - count[0]
    else:
        minority_state = count[0][np.argmin(count[1])]

    # print(minority_state)

    correction_coordinates = []
    for x in range(0, 2*Lx, 2):
        for y in range(0, 2*Ly, 2):
            if state[(x, y)] == minority_state:
                correction_coordinates.append((x, y))

    return correction_coordinates


def get_toric_loop(xcube_matching_ortho, component, proj_axis_int):
    toric_loop_dict = {}
    for loc in xcube_matching_ortho:
        if loc[proj_axis_int] in component:
            loc_2d = tuple_remove(loc, proj_axis_int)

            if loc_2d in toric_loop_dict.keys():
                toric_loop_dict[loc_2d] += 1
            else:
                toric_loop_dict[loc_2d] = 1

    toric_loop = [coord for coord in toric_loop_dict.keys()
                  if toric_loop_dict[coord] % 2 == 1]

    return toric_loop


def tuple_insert(t, index, element):
    new_t = list(t)
    new_t.insert(index, element)
    return tuple(new_t)


def tuple_remove(t, index):
    new_t = list(t)
    new_t.pop(index)
    return tuple(new_t)


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

        Lx, Ly, Lz = code.size

        self.toric_code = {'x': Toric2DCode(Ly, Lz),
                           'y': Toric2DCode(Lx, Lz),
                           'z': Toric2DCode(Lx, Ly)
                           }

        # Weight the 2D toric code matching decoders
        # Only works for Z biased noise and z-axis deformation
        weights_X, _ = self.error_model.get_weights(self.code,
                                                    self.error_rate)

        wz = weights_X[self.code.qubit_index[(0, 0, 1)]]
        wxy = weights_X[self.code.qubit_index[(1, 0, 0)]]

        weights = {
            'x': np.array([wxy if self.toric_code['x'].qubit_axis(loc) == 'x'
                           else wz
                           for loc in self.toric_code['x'].qubit_coordinates]
                          ),
            'y': np.array([wxy if self.toric_code['y'].qubit_axis(loc) == 'x'
                           else wz
                           for loc in self.toric_code['y'].qubit_coordinates]
                          ),
            'z': np.array([wxy
                           for _ in self.toric_code['z'].qubit_coordinates]
                          )}

        self.matching_decoder = {axis: MatchingDecoder(self.toric_code[axis],
                                                       self.error_model,
                                                       self.error_rate,
                                                       weights=(weights[axis],
                                                                weights[axis]))
                                 for axis in ['x', 'y', 'z']}

        self.z_decoder = BeliefPropagationOSDDecoder(self.code,
                                                     self.error_model,
                                                     self.error_rate)

    @property
    def params(self) -> dict:
        return {}

    def decode(self, syndrome: np.ndarray, **kwargs) -> np.ndarray:
        """Get X corrections given code and measured syndrome."""

        # Initialize correction as full bsf.
        possible_correction = {'x': np.zeros(2*self.code.n, dtype=np.uint),
                               'y': np.zeros(2*self.code.n, dtype=np.uint),
                               'z': np.zeros(2*self.code.n, dtype=np.uint)}

        Lx, Ly, Lz = self.code.size

        # print("Initialize plane syndrome")
        plane_syndrome = {
            'x': {x: np.zeros(self.toric_code['x'].n_stabilizers)
                  for x in range(1, 2*Lx, 2)},
            'y': {y: np.zeros(self.toric_code['y'].n_stabilizers)
                  for y in range(1, 2*Ly, 2)},
            'z': {z: np.zeros(self.toric_code['z'].n_stabilizers)
                  for z in range(1, 2*Lz, 2)}
        }

        # Remove X stabilizer syndrome and keep it for later
        x_syndrome = self.code.extract_x_syndrome(syndrome)
        syndrome[self.code.x_indices] = 0
        axis_to_int = {'x': 0, 'y': 1, 'z': 2}

        for proj_axis in ['x', 'y', 'z']:
            proj_axis_int = axis_to_int[proj_axis]

            ortho_axes = tuple_remove(('x', 'y', 'z'), proj_axis_int)

            L_proj = [Lx, Ly, Lz][proj_axis_int]

            for i_stab in range(len(self.code.stabilizer_index)):
                if syndrome[i_stab]:
                    loc = self.code.stabilizer_coordinates[i_stab]

                    for axis in ['x', 'y', 'z']:
                        loc_2d = tuple_remove(loc, axis_to_int[axis])
                        idx_face = self.toric_code[axis].stabilizer_index[
                            loc_2d
                        ]
                        plane = loc[axis_to_int[axis]]
                        plane_syndrome[axis][plane][idx_face] = 1

            # Decode all the 2D toric codes
            xcube_matching: Dict[str, List] = {'x': [], 'y': [], 'z': []}
            connected_planes: Dict[int, set] = {
                plane: set() for plane in plane_syndrome[proj_axis].keys()
            }

            for axis in ['x', 'y', 'z']:
                for plane in plane_syndrome[axis].keys():
                    current_syndrome = plane_syndrome[axis][plane]

                    toric_X_syndrome = self.toric_code[
                        axis
                    ].extract_x_syndrome(
                        current_syndrome
                    )

                    if np.all(toric_X_syndrome == 0):
                        continue

                    toric_matching = self.matching_decoder[axis].decode(
                        current_syndrome
                    )
                    n = self.toric_code[axis].n
                    toric_Z_correction = toric_matching[n:]

                    toric_pairs = get_matched_pairs(
                        self.toric_code[axis].Hx.todense(),
                        toric_Z_correction,
                        toric_X_syndrome
                    )

                    for i in toric_Z_correction.nonzero()[0]:
                        x, y = self.toric_code[axis].qubit_coordinates[i]
                        loc_3d = tuple_insert((x, y), axis_to_int[axis], plane)
                        xcube_matching[axis].append(loc_3d)

                    if axis != proj_axis:
                        for pair in toric_pairs:
                            i1, i2 = pair
                            loc1 = self.toric_code[axis].qubit_coordinates[i1]
                            loc2 = self.toric_code[axis].qubit_coordinates[i2]

                            # print("Axis", axis)
                            # print("Pair", loc1, loc2)

                            proj_component = {'x': {'y': 0, 'z': 0},
                                              'y': {'x': 0, 'z': 1},
                                              'z': {'x': 1, 'y': 1}
                                              }[proj_axis][axis]
                            plane1 = loc1[proj_component]
                            plane2 = loc2[proj_component]

                            if plane1 != plane2:
                                if proj_component == 1:
                                    connected_planes[plane1+1].add(plane2+1)
                                    connected_planes[plane2+1].add(plane1+1)
                                else:
                                    connected_planes[plane1].add(plane2)
                                    connected_planes[plane2].add(plane1)

            xcube_matching_proj = xcube_matching[proj_axis]
            xcube_matching_ortho = set((xcube_matching[ortho_axes[0]]
                                        + xcube_matching[ortho_axes[1]]))

            connected_components = find_connected_components(connected_planes)

            list_plane_proj = []
            for component in list(connected_components):
                plane_proj = component[0]
                list_plane_proj.append(plane_proj)

                # Perform the projection
                for loc in xcube_matching_proj:
                    # print("Z matching", (x, y, z))
                    plane = loc[proj_axis_int]
                    ortho_components = tuple_remove(loc, proj_axis_int)
                    if plane in component:
                        if (0 < plane - plane_proj <= L_proj or
                                2*L_proj + plane - plane_proj <= L_proj):
                            if plane_proj < plane:
                                plane_stop = plane + 1
                            else:
                                plane_stop = 2*L_proj + plane + 1

                            for plane_op in range(plane_proj+1, plane_stop, 2):
                                loc = tuple_insert(ortho_components,
                                                   proj_axis_int,
                                                   plane_op % (2*L_proj))

                                idx = self.code.qubit_index[loc]
                                possible_correction[proj_axis][idx] += 1
                        elif plane != plane_proj:
                            if plane < plane_proj:
                                plane_stop = plane_proj + 1
                            else:
                                plane_stop = 2*L_proj + plane_proj + 1

                            for plane_op in range(plane + 1, plane_stop, 2):
                                loc = tuple_insert(ortho_components,
                                                   proj_axis_int,
                                                   plane_op % (2*L_proj))

                                idx = self.code.qubit_index[loc]
                                possible_correction[proj_axis][idx] += 1

            # Find the loops
            for component in connected_components:
                toric_loop = get_toric_loop(
                    xcube_matching_ortho, component, proj_axis_int
                )
                correction_coordinates = decode_plane(toric_loop, (Lx, Ly))

                plane_proj = component[0]

                for loc_2d in correction_coordinates:
                    loc_3d = tuple_insert(loc_2d, proj_axis_int, plane_proj)
                    idx = self.code.qubit_index[loc_3d]

                    possible_correction[proj_axis][idx] = 1

        weight = [np.sum(c) for c in possible_correction.values()]
        index_min_weight = int(np.argmin(weight))

        axis_min_weight = {0: 'x', 1: 'y', 2: 'z'}[index_min_weight]
        correction = possible_correction[axis_min_weight]

        # Decode Z part with BP-OSD
        syndrome[self.code.z_indices] = 0
        syndrome[self.code.x_indices] = x_syndrome

        z_correction = self.z_decoder.decode(syndrome).astype('uint8')

        correction += z_correction

        return correction % 2
