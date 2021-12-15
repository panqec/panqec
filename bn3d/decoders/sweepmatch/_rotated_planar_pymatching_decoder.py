"""
Decoder for Rotated Planar 3D Code using Pymatching.
"""
from typing import Dict
import numpy as np
from qecsim.model import StabilizerCode
from pymatching import Matching
from ...models import ToricCode3D
from ._pymatching_decoder import Toric3DPymatchingDecoder


class RotatedPlanarPymatchingDecoder(Toric3DPymatchingDecoder):
    """Pymatching decoder for decoding point sector of Rotated Planar 3D Code.

    Can decode multiple different codes.
    """

    label = 'Rotated Planar 3D Pymatching'
    _matchers: Dict[str, Matching] = {}
    _n_faces: Dict[str, int] = {}
    _n_qubits: Dict[str, int] = {}

    def __init__(self):
        self._matchers = {}
        self._n_faces = {}
        self._n_qubits = {}

    def new_matcher(self, code: StabilizerCode):
        """Return a new Matching object."""
        self._n_faces[code.label] = code.get_face_X_stabilizers().shape[0]
        n_qubits = len(code.qubit_index)
        self._n_qubits[code.label] = n_qubits
        return Matching(code.Hx[:, :n_qubits])

    def get_vertex_syndromes(
        self, code: ToricCode3D, full_syndrome: np.ndarray
    ) -> np.ndarray:
        """Get only the syndromes for the vertex Z stabilizers.

        X face stabiziliers syndromes are discarded for this decoder.
        """
        n_faces = self._n_faces[code.label]
        vertex_syndromes = full_syndrome[n_faces:]
        return vertex_syndromes

    def decode(self, code: ToricCode3D, syndrome: np.ndarray) -> np.ndarray:
        """Get X corrections given code and measured syndrome."""

        # Initialize correction as full bsf.
        correction = np.zeros(2*code.n_k_d[0], dtype=np.uint)

        # Get the Pymatching Matching object.
        matcher = self.get_matcher(code)

        # Keep only the vertex Z measurement syndrome, discard the rest.
        vertex_syndromes = self.get_vertex_syndromes(code, syndrome)

        # PyMatching gives only the X correction.
        x_correction = matcher.decode(vertex_syndromes, num_neighbours=None)

        # Load it into the X block of the full bsf.
        n_qubits = self._n_qubits[code.label]
        correction[:n_qubits] = x_correction

        return correction
