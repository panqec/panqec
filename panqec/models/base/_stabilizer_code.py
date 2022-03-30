from typing import Dict, Tuple, Optional, List
from abc import ABCMeta, abstractmethod
import numpy as np
from ...bpauli import bcommute
from ... import bsparse
from scipy.sparse import csr_matrix

Operator = Dict[Tuple, str]  # Coordinate to pauli ('X', 'Y' or 'Z')


class StabilizerCode(metaclass=ABCMeta):
    """Abstract class for generic stabilizer codes (CSS or not)

    Any subclass should override the following four methods:
    - get_qubit_coordinates() to define all the coordinates in the lattice
    that contain qubits
    - get_vertex_coordinates() to define all the coordinates in the lattice
    that contain vertices (could also be another type of stabilizer)
    - get_face_coordinates() to define all the coordinates in the lattice
    that contain faces (could also be another type of stabilizer)
    - axis(location) to return the axis of a qubit at a given location (when qubit
    have an orientation in space, for instance when they are edges)

    Using only those methods, a StabilizerCode will then automatically create the
    corresponding parity-check matrix (in self.stabilizers) and can be used to make
    a visualization in the GUI or calculate thresholds.
    """

    X_AXIS = 0
    Y_AXIS = 1
    Z_AXIS = 2

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

        self._qubit_coordinates = []
        self._stabilizer_coordinates = []

        self._qubit_index = {}
        self._stabilizer_index = {}

        self._stabilizer_matrix = bsparse.empty_row(2*self.n)
        self._Hx = bsparse.empty_row(self.n)
        self._Hz = bsparse.empty_row(self.n)
        self._logicals_x = bsparse.empty_row(2*self.n)
        self._logicals_z = bsparse.empty_row(2*self.n)
        self._is_css = None
        self._x_indices = None
        self._z_indices = None

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
        return len(self.qubit_coordinates)

    @property
    def k(self) -> int:
        """Number of logical qubits"""
        return self.logicals_x.shape[0]

    @property
    def d(self) -> int:
        """Distance of the code"""
        return min(self.logicals_z.shape[1], self.logicals_x.shape[1])

    @property
    def qubit_coordinates(self) -> List[Tuple]:
        """List of all the coordinates that contain a qubit"""

        if len(self._qubit_coordinates) == 0:
            self._qubit_coordinates = self.get_qubit_coordinates()

        return self._qubit_coordinates

    @property
    def stabilizer_coordinates(self) -> List[Tuple]:
        """List of all the coordinates that contain a stabilizer"""

        if len(self._stabilizer_coordinates) == 0:
            self._stabilizer_coordinates = self.get_stabilizer_coordinates()

        return self._stabilizer_coordinates

    @property
    def qubit_index(self) -> Dict[Tuple, int]:
        """Dictionary that assigns an index to a given qubit location"""

        if len(self._qubit_index) == 0:
            self._qubit_index = {loc: i for i, loc in enumerate(self.qubit_coordinates)}

        return self._qubit_index

    @property
    def stabilizer_index(self) -> Dict[Tuple, int]:
        """Dictionary that assigns an index to a given stabilizer location"""

        if len(self._stabilizer_index) == 0:
            self._stabilizer_index = {loc: i for i, loc in enumerate(self.stabilizer_coordinates)}

        return self._stabilizer_index

    @property
    def n_stabilizers(self) -> int:
        """Number of stabilizer generators"""
        return len(self.stabilizer_index)

    @property
    def logicals_x(self) -> csr_matrix:
        """Logical X operator, as a k x 2n sparse matrix in the binary symplectic format,
        where k is the number of logical X operators, and n the number of qubits.
        """
        if self._logicals_x.size == 0:
            for logical_op in self.get_logicals_x():
                self._logicals_x = bsparse.vstack([self._logicals_x, self.to_bsf(logical_op)])

        return self._logicals_x

    @property
    def logicals_z(self) -> csr_matrix:
        """Logical Z operators in the binary symplectic format.
        It is a sparse matrix of dimension k x 2n, where k is the number
        of Z logicals and n the number of qubits.
        """
        if self._logicals_z.size == 0:
            for logical_op in self.get_logicals_z():
                self._logicals_z = bsparse.vstack([self._logicals_z, self.to_bsf(logical_op)])

        return self._logicals_z

    @property
    def is_css(self) -> bool:
        """Determines if a code is CSS, i.e. if it has separate X
        and Z stabilizers
        """
        if self._is_css is None:
            self._is_css = not np.any(np.logical_and(self.x_indices, self.z_indices))

        return self._is_css

    @property
    def stabilizer_matrix(self) -> csr_matrix:
        """Parity-check matrix of the code in the binary symplectic format.
        It is a sparse matrix of dimension k x 2n, where k is the total number
        of stabilizers and n the number of qubits
        """

        if bsparse.is_empty(self._stabilizer_matrix):
            for stabilizer_location in self.stabilizer_index:
                stabilizer_op = self.get_stabilizer(stabilizer_location, deformed_axis=self._deformed_axis)
                self._stabilizer_matrix = bsparse.vstack([self._stabilizer_matrix, self.to_bsf(stabilizer_op)])

        return self._stabilizer_matrix

    @property
    def size(self) -> Tuple:
        """Dimensions of the lattice."""
        return self._size

    @property
    def Hx(self) -> csr_matrix:
        """Parity-check matrix corresponding to the Z stabilizers of the code.
        It is a sparse matrix of dimension k x n, where k is the number of
        Z stabilizers and n the number of qubits.
        Works only for CSS codes.
        """

        if not self.is_css:
            raise ValueError("Impossible to extract Hz: the code is not CSS")

        if self._Hx.shape[0] == 0:
            H = self.stabilizer_matrix[:, :self.n]
            self._Hx = H[self.x_indices]

        return self._Hx

    @property
    def Hz(self) -> csr_matrix:
        """Parity-check matrix corresponding to the Z stabilizers of the code.
        It is a sparse matrix of dimension k x n, where k is the number of
        Z stabilizers and n the number of qubits.
        Works only for CSS codes.
        """

        if not self.is_css:
            raise ValueError("Impossible to extract Hz: the code is not CSS")

        if self._Hz.shape[0] == 0:
            H = self.stabilizer_matrix[:, self.n:]
            self._Hz = H[self.z_indices]

        return self._Hz

    @property
    def x_indices(self) -> np.ndarray:
        """Indices of the X stabilizers in the parity-check matrix,
        as a boolean array s.t. x_indices[i] is True if stabilizer H[i]
        only contain X operators and False otherwise"""

        if self._x_indices is None:
            Hx = self.stabilizer_matrix[:, :self.n]
            self._x_indices = (Hx.getnnz(1) > 0)

        return self._x_indices

    @property
    def z_indices(self) -> np.ndarray:
        """Indices of the Z stabilizers in the parity-check matrix,
        as a boolean array s.t. z_indices[i] is True if stabilizer H[i]
        only contain Z operators and False otherwise"""

        if self._z_indices is None:
            Hz = self.stabilizer_matrix[:, self.n:]
            self._z_indices = (Hz.getnnz(1) > 0)

        return self._z_indices

    def extract_x_syndrome(self, syndrome: np.ndarray) -> np.ndarray:
        """For CSS codes. Returns the part of the syndrome that corresponds to X stabilizers"""

        return syndrome[self.x_indices]

    def extract_z_syndrome(self, syndrome: np.ndarray) -> np.ndarray:
        """For CSS codes. Returns the part of the syndrome that corresponds to Z stabilizers"""

        return syndrome[self.z_indices]

    def to_bsf(self, operator: Operator) -> csr_matrix:
        bsf_operator = bsparse.zero_row(2*self.n)

        for qubit_location in operator.keys():
            if operator[qubit_location] in ['X', 'Y']:
                bsparse.insert_mod2(self.qubit_index[qubit_location], bsf_operator)
            if operator[qubit_location] in ['Y', 'Z']:
                bsparse.insert_mod2(self.n + self.qubit_index[qubit_location], bsf_operator)

        return bsf_operator

    def from_bsf(self, bsf_operator: csr_matrix) -> Operator:
        assert bsf_operator.shape[0] == 1

        operator = dict()

        rows, cols = bsf_operator.nonzero()

        for col in cols:
            if col < self.n:
                operator[self.qubit_coordinates[col]] = 'X'
            else:
                if self.qubit_coordinates[col] in operator.keys():
                    operator[self.qubit_coordinates[col]] = 'Y'
                else:
                    operator[self.qubit_coordinates[col]] = 'Z'

        return operator

    def measure_syndrome(self, error) -> np.ndarray:
        """Perfectly measure syndromes given Pauli error."""
        return bcommute(self.stabilizer_matrix, self.to_bsf(error))

    def is_stabilizer(self, location: Tuple, stab_type: str = None):
        """Returns whether a given location in the coordinate system
        corresponds to a stabilizer or not
        """
        _is_stabilizer = (location in self.stabilizer_index) and\
                         (stab_type is None or self.stabilizer_type(location) == stab_type)

        return _is_stabilizer

    def is_qubit(self, location: Tuple):
        """Returns whether a given location in the coordinate system
        corresponds to a qubit or not
        """
        return location in self.qubit_index

    @abstractmethod
    def get_qubit_coordinates(self) -> List[Tuple]:
        """Create qubit indices.
        Should return a dictionary that assigns an index to a given qubit coordinate.
        It can be constructed by first creating a list of coordinates (all the locations
        in a coordinate system that contain a qubit) and then converting it to a dictionary
        with the correct format
        """
        raise NotImplementedError

    @abstractmethod
    def get_stabilizer_coordinates(self) -> List[Tuple]:
        """Create stabilizer indices.
        Should return a dictionary that assigns an index to a given stabilizer coordinate.
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

    @abstractmethod
    def stabilizer_type(self, location) -> str:
        """ Return the type of a stabilizer sitting at `location`.
        E.g. 'vertex' or 'face' in toric codes
        """
        raise NotImplementedError

    @abstractmethod
    def get_stabilizer(self, location, deformed_axis=None) -> Operator:
        """ Returns a stabilizer, formatted as dictionary that assigns a Pauli operator
        ('X', 'Y' or 'Z') to each qubit location in the support of the stabilizer
        E.g. for a vertex stabilizer, `get_stabilizer((1,1)) -> {(1,0): 'X', (0, 1): 'X', (2, 1): 'X', (1, 2): 'X'}`
        """
        raise NotImplementedError

    @abstractmethod
    def get_logicals_x(self) -> Operator:
        """Return the logical X operators as a dictionary that assigns a Pauli operator
        ('X', 'Y' or 'Z') to each qubit location in the support of the logical operator
        It should have dimension k x 2n, with k the number of logicals X
        (i.e. the number of logical qubits) and n the number of qubits
        """
        raise NotImplementedError

    @abstractmethod
    def get_logicals_z(self) -> Operator:
        """Return the logical Z operators as a dictionary that assigns a Pauli operator
        ('X', 'Y' or 'Z') to each qubit location in the support of the logical operator
        It should have dimension k x 2n, with k the number of logicals X
        (i.e. the number of logical qubits) and n the number of qubits
        """
        raise NotImplementedError
