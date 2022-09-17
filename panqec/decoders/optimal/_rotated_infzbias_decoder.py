from typing import Dict, Tuple, List
import numpy as np
from panqec.decoders import BaseDecoder
from pymatching import Matching
from ..sweepmatch._rotated_sweep_decoder import RotatedSweepDecoder3D


class ZMatchingDecoder(RotatedSweepDecoder3D):
    label = 'Rotated Infinite Z Bias Loop Sector Decoder'
    allowed_codes: List[str] = []

    def get_edges_xy(self):
        xy = [
            (x, y) for x, y, z in self.code.stabilizer_coordinates
            if z == 2 and self.code.stabilizer_type((x, y, z)) == 'face'
        ]
        return xy

    def decode(
        self, syndrome: np.ndarray, **kwargs
    ) -> np.ndarray:
        correction: Dict = dict()
        signs = self.get_initial_state(syndrome)

        # 1D pair matching along each vertical lines of horizontal edges.
        for xy in self.get_edges_xy():
            signs = self.decode_vertical_line(
                signs, correction, xy
            )

        # 2D matching on horizontal planes.
        L_z = self.code.size[2]
        for z_plane in range(1, 2*L_z, 2):
            signs = self.match_horizontal_plane(
                signs, correction, z_plane
            )
        return self.code.to_bsf(correction)

    def match_horizontal_plane(
        self, signs: np.ndarray, correction: Dict,
        z_plane: int
    ):
        """Do 2D matching on top and bottom boundary surfaces."""
        face_coordinates = [
            location
            for location in self.code.stabilizer_coordinates
            if self.code.stabilizer_type(location) == 'face'
        ]
        edges = sorted([
            (x, y, z) for x, y, z in self.code.qubit_coordinates
            if z == z_plane
        ])
        faces = sorted([
            (x, y, z) for x, y, z in face_coordinates if z == z_plane
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
            signs[self.code.stabilizer_index[face]]
            for face in faces
        ], dtype=np.uint)

        surface_corrections = matcher.decode(
            surface_syndromes, num_neighbours=None
        )

        new_signs = signs.copy()
        for i_edge in np.where(surface_corrections)[0]:
            location = edges[i_edge]
            self.flip_edge(location, new_signs)
            self.code.site(correction, 'Z', location)

        return new_signs

    def decode_vertical_line(
        self, signs: np.ndarray, correction: Dict,
        xy: Tuple[int, int]
    ):
        """Do 1D matching along a vertical line."""
        L_z = self.code.size[2]
        x, y = xy

        flip_locations = []

        line_faces = [
            (x, y, 2*i + 2)
            for i in range(L_z-1)
        ]
        line_syndromes = np.array([
            signs[self.code.stabilizer_index[face]]
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
            self.flip_edge(location, new_signs)
            self.code.site(correction, 'Z', location)
        return new_signs


class XLineDecoder(BaseDecoder):

    label = 'Rotated Infinite Z Bias Point Sector Decoder'
    allowed_codes: List[str] = []

    def decode_line(
        self, syndrome: np.ndarray, xy: Tuple[int, int]
    ) -> np.ndarray:
        x, y = xy
        L_z = self.code.size[2]

        edges = [
            (x, y, 2*i + 2)
            for i in range(L_z - 1)
        ]
        vertices = [
            (x, y, 2*i + 1)
            for i in range(L_z)
        ]
        edge_index = {
            location: index for index, location in enumerate(edges)
        }

        # Construct the check matrix for PyMatching.
        check_matrix = np.zeros((len(vertices), len(edges)), dtype=np.uint)
        for i_vertex, (x, y, z) in enumerate(vertices):
            neighbouring_edges = [
                (x, y, z + 1),
                (x, y, z - 1),
            ]
            for edge in neighbouring_edges:
                if edge in edges:
                    check_matrix[i_vertex, edge_index[edge]] = 1
        matcher = Matching(check_matrix)

        line_syndromes = np.array([
            syndrome[self.code.stabilizer_index[vertex]]
            for vertex in vertices
        ], dtype=np.uint)

        line_corrections = matcher.decode(
            line_syndromes, num_neighbours=None
        )

        x_correction = np.zeros(self.code.n, dtype=np.uint)
        for i_edge in np.where(line_corrections)[0]:
            x_correction[self.code.qubit_index[edges[i_edge]]] = 1

        return x_correction

    def decode(
        self, syndrome: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Get X corrections given code and measured syndrome."""

        # Initialize correction as full bsf.
        n_qubits = self.code.n
        correction = np.zeros(2*n_qubits, dtype=np.uint)
        x_correction = np.zeros(n_qubits, dtype=np.uint)

        lines_xy = sorted([
            (x, y) for x, y, z in self.code.qubit_index if z == 2
        ])
        for xy in lines_xy:
            x_correction += self.decode_line(syndrome, xy)

        # Load it into the X block of the full bsf.
        correction[:n_qubits] = x_correction

        return correction


class RotatedInfiniteZBiasDecoder(BaseDecoder):
    """An optimal decoder for infinite Z bias on deformed noise."""

    label = 'Rotated Infinite Z Bias Decoder'
    allowed_codes: List[str] = []

    matcher: XLineDecoder
    sweeper: ZMatchingDecoder

    def __init__(self, code, error_model, error_rate):
        super().__init__(code, error_model, error_rate)
        self.matcher = XLineDecoder(code, error_model, error_rate)
        self.sweeper = ZMatchingDecoder(code, error_model, error_rate)

    def decode(
        self, syndrome: np.ndarray, **kwargs
    ) -> np.ndarray:

        z_correction = self.sweeper.decode(syndrome)
        x_correction = self.matcher.decode(syndrome)

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
