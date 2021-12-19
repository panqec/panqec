from typing import Dict, Tuple, Optional
from abc import ABCMeta
import numpy as np
from ._indexed_code import IndexedCode
from ... import sparse

Indexer = Dict[Tuple[int, int, int], int]


class IndexedSparsePauli(metaclass=ABCMeta):

    def __init__(self, code: IndexedCode, bsf: Optional[np.ndarray] = None):

        # Copy needs to be made because numpy arrays are mutable.
        self._code = code
        self._from_bsf(bsf)

    def _from_bsf(self, bsf):
        # initialise lattices for X and Z operators from bsf
        n_qubits = self.code.n_k_d[0]
        # initialise identity lattices for X and Z operators
        self._xs = sparse.zero_row(n_qubits)
        self._zs = sparse.zero_row(n_qubits)

        if bsf is not None:
            assert bsf.shape[1] == 2 * n_qubits, \
                'BSF {} has incompatible length'.format(bsf)
            assert np.all(bsf.data == 1), \
                'BSF {} is not in binary form'.format(bsf)
            # initialise lattices for X and Z operators from bsf

            self._xs, self._zs = sparse.hsplit(bsf)

    def site(self, operator, *indices):
        """
        Apply the operator to site identified by the index.
        Notes:
        * Index is in the format (x, y).
        * Index is modulo lattice dimensions, i.e. on a (2, 2) lattice, (2, -1)
        indexes the same site as (0, 1).
        :param operator: Pauli operator. One of 'I', 'X', 'Y', 'Z'.
        :type operator: str
        :param indices: Any number of indices identifying sites in the format
        (x, y).
        :type indices: Any number of 2-tuple of int
        :return: self (to allow chaining)
        :rtype: RotatedPlanar3DPauli
        """
        for coord in indices:
            # flip sites
            flat_index = self.get_index(coord)
            if operator in ('X', 'Y'):
                sparse.insert_mod2(flat_index, self._xs)
            if operator in ('Z', 'Y'):
                sparse.insert_mod2(flat_index, self._zs)
        return self

    def get_index(self, coordinate):
        coordinate = tuple(coordinate)

        if coordinate not in self.code.qubit_index.keys():
            raise ValueError(
                f"Incorrect qubit coordinate {coordinate} given when "
                "constructing the operator"
            )
        return self.code.qubit_index[coordinate]

    @property
    def code(self):
        """
        The rotated toric code.
        :rtype: RotatedToricCode
        """
        return self._code

    def operator(self, coord):
        """
        Returns the operator on the site identified by the coordinates.
        Notes:
        * coord is in the format (x, y, z).
        :param coord: Coordinate identifying a site in the format (x, y, z).
        :type  coord: 3-tuple of int
        :return: Pauli operator. One of 'I', 'X', 'Y', 'Z'.
        :rtype: str
        """
        # extract binary x and z
        index = self.code.qubit_index[coord]
        x = sparse.is_one(index, self._xs.index)
        z = sparse.is_one(index, self._zs.index)
        # return Pauli
        if x and z:
            return 'Y'
        if x:
            return 'X'
        if z:
            return 'Z'
        else:
            return 'I'

    def copy(self):
        """
        Returns a copy of this Pauli that references the same code but is
        backed by a copy of the bsf.
        :return: A copy of this Pauli.
        :rtype: ToricPauli
        """
        return self.code.pauli_class(self.code, bsf=self.to_bsf())

    def __eq__(self, other):
        if type(other) is type(self):
            return np.array_equal(self._xs.indices, other._xs.indices) and np.array_equal(
                self._zs.indices, other._zs.indices
            )
        return NotImplemented

    def __repr__(self):
        bsf = self.to_bsf()
        return '{}({!r}, {!r}: {!r})'.format(
            type(self).__name__, self.code, bsf.shape, bsf.indices
        )

    def __str__(self):
        """
        ASCII art style lattice showing primal lattice lines and Pauli
        operators.
        :return: Informal string representation.
        :rtype: str
        """
        return self.code.ascii_art(pauli=self)

    def to_bsf(self):
        """
        Binary symplectic representation of Pauli.
        Notes:
        :return: Binary symplectic representation of Pauli in sparse format.
        :rtype: scipy.sparse.csr_matrix (1d)
        """

        return sparse.hstack([self._xs, self._zs])
