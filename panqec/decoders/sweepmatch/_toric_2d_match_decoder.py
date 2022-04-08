from typing import Dict, Tuple
import numpy as np
from pymatching import Matching
from panqec.decoders import BaseDecoder
from panqec.codes import Toric2DCode
import panqec.bsparse as bsparse


class Toric2DPymatchingDecoder(BaseDecoder):
    """Pymatching decoder for 2D Toric Code for unbiased noise."""

    label = 'Toric 2D Pymatching'
    _matchers: Dict[str, Tuple[Matching, Matching]] = {}

    def __init__(self, error_model, probability):
        super().__init__(error_model, probability)
        self._matchers = {}

    def _new_matchers(self, code: Toric2DCode) -> Tuple[Matching, Matching]:
        """Return a new tuple of Matching objects."""

        return Matching(code.Hz), Matching(code.Hx)

    def get_matchers(self, code: Toric2DCode) -> Tuple[Matching, Matching]:
        """Get the matchers for Z and X stabilizers given the code.

        Matching objects are only instantiated once for the same code.
        """

        # Only instantiate a new Matching object if the code hasn't been seen
        # before.
        if code.label not in self._matchers:
            self._matchers[code.label] = self._new_matchers(code)
        return self._matchers[code.label]

    def decode(self, code: Toric2DCode, syndrome: np.ndarray) -> np.ndarray:
        """Get X corrections given code and measured syndrome."""

        # The number of qubits.
        n_qubits = code.n
        n_vertices = code.Hz.shape[0]

        # Initialize correction as full bsf.
        correction = np.zeros(2*n_qubits, dtype=np.uint)

        # Get the Pymatching Matching objects.
        matcher_z, matcher_x = self.get_matchers(code)

        # Keep only the vertex Z measurement syndrome, discard the rest.
        syndromes_z = syndrome[:n_vertices]
        syndromes_x = syndrome[n_vertices:]

        # Match each block using corresponding syndrome but applying correction
        # on the other block.
        correction_z = matcher_z.decode(syndromes_z, num_neighbours=None)
        correction_x = matcher_x.decode(syndromes_x, num_neighbours=None)

        # Load it into the X block of the full bsf.
        correction[n_qubits:] = correction_x
        correction[:n_qubits] = correction_z

        return bsparse.from_array(correction)
