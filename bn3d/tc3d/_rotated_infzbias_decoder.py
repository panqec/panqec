from typing import Dict, Tuple, List
import numpy as np
from qecsim.model import Decoder
from pymatching import Matching
from ._rotated_planar_code_3d import RotatedPlanarCode3D
from ._rotated_planar_3d_pauli import RotatedPlanar3DPauli
from ._rotated_planar_pymatching_decoder import RotatedPlanarPymatchingDecoder
from ._rotated_sweep_decoder import RotatedSweepDecoder3D
Indexer = Dict[Tuple[int, int, int], int]


class ZMatchingDecoder(RotatedSweepDecoder3D):
    def __init__(self):
        pass

    def get_edges_xy(self, code: RotatedPlanarCode3D):
        xy = [
            (x, y) for x, y, z in code.face_index if z == 2
        ]
        return xy

    def decode(
        self, code: RotatedPlanarCode3D, syndrome: np.ndarray
    ) -> np.ndarray:
        correction = RotatedPlanar3DPauli(code)
        signs = self.get_initial_state(code, syndrome)

        # 2D matching on horizontal planes.
        L_z = code.size[2]
        for z_plane in range(1, 2*L_z + 2, 2):
            signs = self.match_horizontal_plane(
                signs, correction, code, z_plane
            )

        # 1D pair matching along each vertical lines of horizontal edges.
        for xy in self.get_edges_xy(code):
            signs = self.decode_vertical_line(
                signs, correction, code, xy
            )
        return correction.to_bsf()

    def match_horizontal_plane(
        self, signs: Indexer, correction: RotatedPlanar3DPauli,
        code: RotatedPlanarCode3D, z_plane: int
    ):
        """Do 2D matching on top and bottom boundary surfaces."""
        edges = sorted([
            (x, y, z) for x, y, z in code.qubit_index if z == z_plane
        ])
        faces = sorted([
            (x, y, z) for x, y, z in code.face_index if z == z_plane
        ])
        edge_index = {
            location: index for index, location in enumerate(edges)
        }

        # Construct the check matrix for PyMatching.
        check_matrix = np.zeros((len(faces), len(edges)), dtype=np.uint)
        for i_face, (x, y, z) in enumerate(faces):
            neighbouring_edges = [
                (x + 1, y + 1, z),
                (x - 1, y + 1, z),
                (x - 1, y - 1, z),
                (x + 1, y - 1, z),
            ]
            for edge in neighbouring_edges:
                if edge in edge_index:
                    check_matrix[i_face, edge_index[edge]] = 1
        matcher = Matching(check_matrix)

        surface_syndromes = np.array([
            signs[face]
            for face in faces
        ], dtype=np.uint)

        surface_corrections = matcher.decode(
            surface_syndromes, num_neighbours=None
        )

        new_signs = signs.copy()
        for i_edge in np.where(surface_corrections)[0]:
            location = edges[i_edge]
            print(location)
            self.flip_edge(location, new_signs, code)
            correction.site('Z', location)

        return new_signs

    def decode_vertical_line(
        self, signs: Indexer, correction: RotatedPlanar3DPauli,
        code: RotatedPlanarCode3D, xy: Tuple[int, int]
    ):
        """Do 1D matching along a vertical line."""
        L_z = code.size[2]
        x, y = xy

        flip_locations = []

        line_faces = [
            (x, y, 2*i + 2)
            for i in range(L_z)
        ]
        line_syndromes = np.array([
            signs[face]
            for face in line_faces
        ])
        n_syndromes = sum(line_syndromes)
        if n_syndromes:
            syndrome_locations = np.sort(np.where(line_syndromes)[0]).tolist()
            segments = split_posts_at_active_fences(
                active_fences=syndrome_locations,
                n_fences=L_z
            )
            segments_even = segments[::2]
            segments_odd = segments[1::2]
            len_segments_even = sum(map(len, segments_even))
            len_segments_odd = sum(map(len, segments_odd))
            if len_segments_even <= len_segments_odd:
                chosen_segments = segments_even
            else:
                chosen_segments = segments_odd
            flip_sites = [i for segment in chosen_segments for i in segment]

            for flip_site in flip_sites:
                z = 2*flip_site + 1
                flip_locations.append((x, y, z))

        new_signs = signs.copy()
        for location in flip_locations:
            self.flip_edge(location, new_signs, code)
            correction.site('Z', location)
        return new_signs


class RotatedInfiniteZBiasDecoder(Decoder):
    """An optimal decoder for infinite Z bias on deformed noise."""

    label = 'Rotated Infinite Z Bias Decoder'
    _matcher: RotatedPlanarPymatchingDecoder
    _sweeper: ZMatchingDecoder

    def __init__(self):
        self._matcher = RotatedPlanarPymatchingDecoder()
        self._sweeper = ZMatchingDecoder()

    def decode(
        self, code: RotatedPlanarCode3D, syndrome: np.ndarray
    ) -> np.ndarray:

        z_correction = self._sweeper.decode(code, syndrome)
        x_correction = self._matcher.decode(code, syndrome)

        correction = (z_correction + x_correction) % 2
        correction = correction.astype(np.uint)
        return correction


def split_posts_at_active_fences(
    active_fences: list, n_fences: int
) -> List[List[int]]:
    posts = list(range(n_fences + 1))
    segments = []
    segment = []
    for post in posts:
        segment.append(post)
        if post in active_fences:
            segments.append(segment)
            segment = []
    segments.append(segment)
    return segments
