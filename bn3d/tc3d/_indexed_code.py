from typing import Dict, Tuple, Optional
from abc import ABCMeta, abstractmethod
import numpy as np
from qecsim.model import StabilizerCode
from bn3d.bpauli import bcommute
Indexer = Dict[Tuple[int, int, int], int]


class IndexedCode(StabilizerCode, metaclass=ABCMeta):

    _size: Tuple[int, int, int]
    _qubit_index: Indexer
    _vertex_index: Indexer
    _face_index: Indexer
    _stabilizers = np.array([])
    _Hx = np.array([])
    _Hz = np.array([])
    _logical_xs = np.array([])
    _logical_zs = np.array([])

    def __init__(
        self, L_x: int,
        L_y: Optional[int] = None,
        L_z: Optional[int] = None
    ):
        if L_y is None:
            L_y = L_x
        if L_z is None:
            L_z = L_x

        self._size = (L_x, L_y, L_z)
        self._qubit_index = self._create_qubit_indices()
        self._vertex_index = self._create_vertex_indices()
        self._face_index = self._create_face_indices()

    @property
    @abstractmethod
    def pauli_class(self):
        """The Pauli operator class."""

    @property
    def qubit_index(self) -> Indexer:
        return self._qubit_index

    @property
    def vertex_index(self) -> Indexer:
        return self._vertex_index

    @property
    def face_index(self) -> Indexer:
        return self._face_index

    @abstractmethod
    def _create_qubit_indices(self) -> Indexer:
        """Create qubit indices."""

    @abstractmethod
    def _create_vertex_indices(self) -> Indexer:
        """Create vertex indices."""

    @abstractmethod
    def _create_face_indices(self):
        """Create face indices."""

    @property
    def stabilizers(self) -> np.ndarray:
        if self._stabilizers.size == 0:
            face_stabilizers = self.get_face_X_stabilizers()
            vertex_stabilizers = self.get_vertex_Z_stabilizers()
            self._stabilizers = np.concatenate([
                face_stabilizers,
                vertex_stabilizers,
            ])
        return self._stabilizers

    @property
    def size(self) -> Tuple[int, int, int]:
        """Dimensions of lattice."""
        return self._size

    @property
    def Hz(self) -> np.ndarray:
        if self._Hz.size == 0:
            self._Hz = self.get_face_X_stabilizers()
        return self._Hz[:, :self.n_k_d[0]]

    @property
    def Hx(self) -> np.ndarray:
        if self._Hx.size == 0:
            self._Hx = self.get_vertex_Z_stabilizers()
        return self._Hx[:, self.n_k_d[0]:]

    def get_vertex_Z_stabilizers(self) -> np.ndarray:
        vertex_stabilizers = []

        for (x, y, z) in self.vertex_index.keys():
            operator = self.pauli_class(self)
            operator.vertex('Z', (x, y, z))
            vertex_stabilizers.append(operator.to_bsf())

        return np.array(vertex_stabilizers, dtype=np.uint)

    def get_face_X_stabilizers(self) -> np.ndarray:
        face_stabilizers = []

        for (x, y, z) in self.face_index.keys():
            operator = self.pauli_class(self)
            operator.face('X', (x, y, z))
            face_stabilizers.append(operator.to_bsf())

        return np.array(face_stabilizers, dtype=np.uint)

    def measure_syndrome(self, error) -> np.ndarray:
        """Perfectly measure syndromes given Pauli error."""
        return bcommute(self.stabilizers, error.to_bsf())


class IndexedCodePauli(metaclass=ABCMeta):

    def __init__(self, code: IndexedCode, bsf: Optional[np.ndarray] = None):

        # Copy needs to be made because numpy arrays are mutable.
        self._code = code
        self._from_bsf(bsf)

    def _from_bsf(self, bsf):
        # initialise lattices for X and Z operators from bsf
        n_qubits = self.code.n_k_d[0]
        if bsf is None:
            # initialise identity lattices for X and Z operators
            self._xs = np.zeros(n_qubits, dtype=int)
            self._zs = np.zeros(n_qubits, dtype=int)
        else:
            assert len(bsf) == 2 * n_qubits, \
                'BSF {} has incompatible length'.format(bsf)
            assert np.array_equal(bsf % 2, bsf), \
                'BSF {} is not in binary form'.format(bsf)
            # initialise lattices for X and Z operators from bsf
            self._xs, self._zs = np.hsplit(bsf, 2)  # split out Xs and Zs

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
                self._xs[flat_index] ^= 1
            if operator in ('Z', 'Y'):
                self._zs[flat_index] ^= 1
        return self

    def get_index(self, coordinate):
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
        x = self._xs[index]
        z = self._zs[index]
        # return Pauli
        if x == 1 and z == 1:
            return 'Y'
        if x == 1:
            return 'X'
        if z == 1:
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
        return self.code.new_pauli(bsf=np.copy(self.to_bsf()))

    def __eq__(self, other):
        if type(other) is type(self):
            return np.array_equal(self._xs, other._xs) and np.array_equal(
                self._zs, other._zs
            )
        return NotImplemented

    def __repr__(self):
        return '{}({!r}, {!r})'.format(
            type(self).__name__, self.code, self.to_bsf()
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
        * For performance reasons, the returned bsf is a view of this Pauli.
        Modifying one will modify the other.
        :return: Binary symplectic representation of Pauli.
        :rtype: numpy.array (1d)
        """
        return np.concatenate((self._xs.flatten(), self._zs.flatten()))
