from typing import Dict, Tuple, Optional
from abc import ABCMeta, abstractmethod
import numpy as np
from ...bpauli import bcommute
from ... import bsparse
from ._stabilizer_pauli import StabilizerPauli
from scipy.sparse import csr_matrix

Indexer = Dict[Tuple, int]  # coordinate to index


class StabilizerCode(metaclass=ABCMeta):
    """Abstract class for generic stabilizer codes (CSS or not)

    Any subclass should override the following four methods:
    - _create_qubit_indices() to define all the coordinates in the lattice
    that contain qubits
    - _create_vertex_indices() to define all the coordinates in the lattice
    that contain vertices (could also be another type of stabilizer)
    - _create_face_indices() to define all the coordinates in the lattice
    that contain faces (could also be another type of stabilizer)
    - axis(location) to return the axis of a qubit at a given location (when qubit
    have an orientation in space, for instance when they are edges)
    - pauli_class(), which returns a subclass of StabilizerPauli that defines how
    stabilizers are constructed for our code

    Using only those methods, a StabilizerCode will then automatically create the
    corresponding parity-check matrix (in self.stabilizers) and can be used to make
    a visualization in the GUI or calculate thresholds.
    """

    X_AXIS = 0
    Y_AXIS = 1
    Z_AXIS = 2

    _size: np.ndarray  # dimensions of the lattice
    _qubit_index: Indexer  # Qubit coordinate to index
    _vertex_index: Indexer  # Vertex coordinate to index
    _face_index: Indexer  # Face coordinate to index
    _stabilizers = np.array([])  # Complete parity-check matrix
    _Hx = np.array([])  # Parity-check matrix for X stabilizers
    _Hz = np.array([])  # Parity-check matrix for Z stabilizers
    _logical_xs = np.array([])  # Parity-check matrix for Z stabilizers
    _logical_zs = np.array([])

    def __init__(
        self, L_x: int,
        L_y: Optional[int] = None,
        L_z: Optional[int] = None,
        deformed_axis: Optional[int] = None
    ):
        """Constructor for the StabilizerCode class

        Parameters
        ----------
        L_x : int
            Dimension of the lattice in the x direction (or in all directions
            if L_y and L_z are not given)
        L_y: int, optional
            Dimension of the lattice in the y direction
        L_z: int, optional
            Dimension of the lattice in the z direction
        deformed_axis: int, optional
            If given, will determine whether to apply a Clifford deformation on this axis.
            The axis is a number between 0 and d, where d is the dimension of the code.
            Can be used to easily create codes such as the XZZX surface code (arXiv: 2009.07851)
        """

        if L_y is None:
            L_y = L_x
        if L_z is None:
            L_z = L_x

        self._deformed_axis = deformed_axis

        if self.dimension == 2:
            self._size = (L_x, L_y)
        else:
            self._size = (L_x, L_y, L_z)

        self._qubit_index = self._create_qubit_indices()
        self._vertex_index = self._create_vertex_indices()
        self._face_index = self._create_face_indices()

        self._stabilizers = bsparse.empty_row(2*self.n)
        self._Hx = bsparse.empty_row(self.n)
        self._Hz = bsparse.empty_row(self.n)
        self._logical_xs = bsparse.empty_row(self.n)
        self._logical_zs = bsparse.empty_row(self.n)

    @property
    @abstractmethod
    def pauli_class(self) -> StabilizerPauli:
        """The Pauli operator class, used to construct the stabilizers."""
        raise NotImplementedError

    @property
    @abstractmethod
    def label(self) -> str:
        """Label uniquely identifying a code, including its lattice dimensions
        Example: 'Toric 3D {Lx}x{Ly}x{Lz}'
        """
        raise NotImplementedError

    @property
    def id(self) -> str:
        """Returns a string identifying the class (usually the code name)"""
        return self.__class__.__name__

    @property
    def n(self) -> int:
        """Number of physical qubits"""
        return len(self.qubit_index)

    @property
    def k(self) -> int:
        """Number of logical qubits"""
        return len(self.logical_xs)

    @property
    def d(self) -> int:
        """Distance of the code"""
        return min(self.logical_zs.shape[1], self.logical_xs.shape[1])

    @property
    def qubit_index(self) -> Indexer:
        """Returns a dictionary that assigns an index to a given qubit coordinate"""
        return self._qubit_index

    @property
    def vertex_index(self) -> Indexer:
        """Returns a dictionary that assigns an index to a given vertex coordinate"""
        return self._vertex_index

    @property
    def face_index(self) -> Indexer:
        """Returns a dictionary that assigns an index to a given face coordinate"""
        return self._face_index

    @property
    def n_faces(self) -> int:
        """Return the number of face stabilizers"""
        return len(self.face_index)

    @property
    def n_vertices(self) -> int:
        """Return the number of vertex stabilizers"""
        return len(self.vertex_index)

    @abstractmethod
    def _create_qubit_indices(self) -> Indexer:
        """Create qubit indices.
        Should return a dictionary that assigns an index to a given qubit coordinate.
        It can be constructed by first creating a list of coordinates (all the locations
        in a coordinate system that contain a qubit) and then converting it to a dictionary
        with the correct format
        """
        raise NotImplementedError

    @abstractmethod
    def _create_vertex_indices(self) -> Indexer:
        """Create vertex indices.
        Should return a dictionary that assigns an index to a given qubit coordinate.
        It can be constructed by first creating a list of coordinates (all the locations
        in a coordinate system that contain a qubit) and then converting it to a dictionary
        with the correct format
        """
        raise NotImplementedError

    @abstractmethod
    def _create_face_indices(self):
        """Create face indices.
        Should return a dictionary that assigns an index to a given qubit coordinate.
        It can be constructed by first creating a list of coordinates (all the locations
        in a coordinate system that contain a qubit) and then converting it to a dictionary
        with the correct format
        """
        raise NotImplementedError

    @abstractmethod
    def axis(self, location) -> int:
        """ Return the axis of a qubit sitting at given location.
        Useful when qubits have an orientation in space, for instance when they are edges,
        to simplify the construction of stabilizers and the Clifford deformations
        """
        raise NotImplementedError

    @property
    def stabilizers(self) -> csr_matrix:
        """Returns the parity-check matrix of the code.
        It is a sparse matrix of dimension k x 2n, where k is the number of stabilizers
        and n the number of qubits
        """

        if bsparse.is_empty(self._stabilizers):
            face_stabilizers = self.get_face_stabilizers()
            vertex_stabilizers = self.get_vertex_stabilizers()
            self._stabilizers = bsparse.vstack([
                face_stabilizers,
                vertex_stabilizers,
            ])
        return self._stabilizers

    @property
    @abstractmethod
    def logical_xs(self) -> csr_matrix:
        """Return the logical X operators in the sparse binary symplectic format
        It should have dimension k x 2n, with k the number of logicals X
        (i.e. the number of logical qubits) and n the number of qubits
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def logical_zs(self) -> csr_matrix:
        """Return the logical Z operators in the sparse binary symplectic format
        It should have dimension k x 2n, with k the number of logicals X
        (i.e. the number of logical qubits) and n the number of qubits
        """
        raise NotImplementedError

    @property
    def size(self) -> Tuple:
        """Dimensions of lattice."""
        return self._size

    @property
    def Hz(self) -> csr_matrix:
        """Returns a parity-check matrix of dimension k x n (in sparse format)
        where k is the number of face stabilizers and n the number of qubits.
        Useful only for CSS codes, where face stabilizers only contain X operators
        """
        if self._Hz.shape[0] == 0:
            self._Hz = self.stabilizers[:self.n_faces, :self.n]
        return self._Hz

    @property
    def Hx(self) -> csr_matrix:
        """Returns a parity-check matrix of dimension k x n (in sparse format)
        where k is the number of face stabilizers and n the number of qubits.
        Useful only for CSS codes, where face stabilizers only contain Z operators
        """
        if self._Hx.shape[0] == 0:
            self._Hx = self.stabilizers[self.n_faces:, self.n:]
        return self._Hx

    def get_vertex_stabilizers(self):
        """Returns the parity-check matrix of vertex stabilizers
        It is a sparse matrix of dimension k x 2n, where k is the number of vertex
        stabilizers and n the number of qubits.
        Note that it can contain both X and Z paulis
        """
        vertex_stabilizers = bsparse.empty_row(2*self.n)

        for vertex_location in self.vertex_index.keys():
            operator = self.pauli_class(self)
            operator.vertex('Z', vertex_location, deformed_axis=self._deformed_axis)
            vertex_stabilizers = bsparse.vstack([vertex_stabilizers, operator.to_bsf()])

        return vertex_stabilizers

    def get_face_stabilizers(self):
        """Returns the parity-check matrix of face stabilizers
        It is a sparse matrix of dimension k x 2n, where k is the number of face
        stabilizers and n the number of qubits.
        Note that it can contain both X and Z paulis
        """
        face_stabilizers = bsparse.empty_row(2*self.n)

        for face_location in self.face_index.keys():
            operator = self.pauli_class(self)
            operator.face('X', face_location, deformed_axis=self._deformed_axis)
            face_stabilizers = bsparse.vstack([face_stabilizers, operator.to_bsf()])

        return face_stabilizers

    def measure_syndrome(self, error) -> np.ndarray:
        """Perfectly measure syndromes given Pauli error."""
        return bcommute(self.stabilizers, error.to_bsf())

    def is_face(self, location: Tuple) -> bool:
        """Returns whether a given location in the coordinate system
        corresponds to a face or not
        """
        return location in self.face_index.keys()

    def is_vertex(self, location):
        """Returns whether a given location in the coordinate system
        corresponds to a vertex or not
        """
        return location in self.vertex_index.keys()

    def is_qubit(self, location):
        """Returns whether a given location in the coordinate system
        corresponds to a qubit or not
        """
        return location in self.qubit_index.keys()
