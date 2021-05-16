"""
Classes for 2D Toric Codes
"""

from typing import Dict, Tuple
import numpy as np
from pymatching import Matching
from qecsim.model import Decoder
from qecsim.models.toric import ToricCode


class Toric2DPymatchingDecoder(Decoder):
    """Pymatching decoder for 2D Toric Code for unbiased noise."""

    label = 'Toric 2D Pymatching'
    _matchers: Dict[str, Tuple[Matching, Matching]] = {}

    def __init__(self):
        self._matchers = {}

    def _new_matchers(self, code: ToricCode) -> Tuple[Matching, Matching]:
        """Return a new tuple of Matching objects."""
        # Get the number of X stabilizers (faces).
        n_vertices = int(np.product(code.size))
        n_qubits = code.n_k_d[0]

        # Matcher for Z stabilizers to detect X errors.
        H_z = code.stabilizers[:n_vertices, n_qubits:]

        # Matcher for X stabilizers to detect Z errors.
        H_x = code.stabilizers[n_vertices:, :n_qubits]
        return Matching(H_z), Matching(H_x)

    def get_matchers(self, code: ToricCode) -> Tuple[Matching, Matching]:
        """Get the matchers for Z and X stabilizers given the code.

        Matching objects are only instantiated once for the same code.
        """

        # Only instantiate a new Matching object if the code hasn't been seen
        # before.
        if code.label not in self._matchers:
            self._matchers[code.label] = self._new_matchers(code)
        return self._matchers[code.label]

    def decode(self, code: ToricCode, syndrome: np.ndarray) -> np.ndarray:
        """Get X corrections given code and measured syndrome."""

        # The number of qubits.
        n_qubits = code.n_k_d[0]
        n_vertices = int(np.product(code.size))

        # Initialize correction as full bsf.
        correction = np.zeros(2*n_qubits, dtype=np.uint)

        # Get the Pymatching Matching objects.
        matcher_z, matcher_x = self.get_matchers(code)

        # Keep only the vertex Z measurement syndrome, discard the rest.
        syndromes_z = syndrome[:n_vertices]
        syndromes_x = syndrome[n_vertices:]

        # Match each block using corresponding syndrome but applying correction
        # on the other block.
        correction_x = matcher_z.decode(syndromes_z, num_neighbours=None)
        correction_z = matcher_x.decode(syndromes_x, num_neighbours=None)

        # Load it into the X block of the full bsf.
        correction[:n_qubits] = correction_x
        correction[n_qubits:] = correction_z

        return correction
