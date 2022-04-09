from typing import Dict, Tuple, Optional, List
import os
from abc import ABCMeta, abstractmethod
import numpy as np
import json
from scipy.sparse import csr_matrix, dok_matrix

import panqec
from panqec.bpauli import bcommute
from panqec import bsparse

os.environ['PANQEC_ROOT_DIR'] = os.path.dirname(panqec.__file__)

Operator = Dict[Tuple, str]  # Coordinate to pauli ('X', 'Y' or 'Z')


class StabilizerCode(metaclass=ABCMeta):
    """Abstract class for generic stabilizer codes (CSS or not)

    Any subclass should override the following four methods:
    - get_qubit_coordinates() to define all the coordinates in the lattice
    that contain qubits
    - get_stabilizer_coordinates() to define all the coordinates in the lattice
    that contain stabilizers
    - qubit_axis(location) to return the axis of a qubit at a given location (when qubit
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
        deformed_axis: Optional[str] = None
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
        deformed_axis: str, optional
            If given, will determine whether to apply a Clifford deformation on this axis.
            The axis is a string in ['x', 'y', 'z'].
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

        self.colormap = {'red': '0xFF4B3E',
                         'blue': '0x48BEFF',
                         'green': '0x058C42',
                         'pink': '0xffbcbc',
                         'white': '0xf2f2fc',
                         'gold': '0xf1c232',
                         'coral': '0xFA824C',
                         'light-yellow': '0xFAFAC6',
                         'salmon': '0xe79e90',
                         'light-orange': '0xFA824C',
                         'orange': '0xfa7921'}

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimension of the code (usually 2 or 3)"""

    @property
    @abstractmethod
    def label(self) -> str:
        """Label uniquely identifying a code, including its lattice dimensions
        Example: 'Toric 3D {Lx}x{Ly}x{Lz}'
        """

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
        """Logical X operator, as a k x 2n sparse matrix in the binary
        symplectic format, where k is the number of logical X operators,
        and n the number of qubits.
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
            sparse_dict = {}
            self._stabilizer_matrix = dok_matrix((self.n_stabilizers, 2*self.n))

            for i_stab, stabilizer_location in enumerate(self.stabilizer_coordinates):
                stabilizer_op = self.get_stabilizer(stabilizer_location, deformed_axis=self._deformed_axis)

                for qubit_location in stabilizer_op.keys():
                    if stabilizer_op[qubit_location] in ['X', 'Y']:
                        i_qubit = self.qubit_index[qubit_location]
                        if (i_stab, i_qubit) in sparse_dict.keys():
                            sparse_dict[(i_stab, i_qubit)] += 1
                        else:
                            sparse_dict[(i_stab, i_qubit)] = 1
                    if stabilizer_op[qubit_location] in ['Y', 'Z']:
                        i_qubit = self.n + self.qubit_index[qubit_location]
                        if (i_stab, i_qubit) in sparse_dict.keys():
                            sparse_dict[(i_stab, i_qubit)] += 1
                        else:
                            sparse_dict[(i_stab, i_qubit)] = 1

            self._stabilizer_matrix._update(sparse_dict)
            self._stabilizer_matrix = self._stabilizer_matrix.tocsr()
            self._stabilizer_matrix.data %= 2

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

    def extract_x_syndrome(self, syndrome: csr_matrix) -> csr_matrix:
        """For CSS codes only. Returns the part of the syndrome that
        corresponds to X stabilizers.

        Parameters
        ----------
        syndrome: csr_matrix
            Syndrome as a sparse row of dimension 1xm, where m is the number
            of stabilizers.

        Returns
        -------
        x_syndrome: csr_matrix
            Syndrome reduced to X stabilizers
        """

        return syndrome[self.x_indices]

    def extract_z_syndrome(self, syndrome: csr_matrix) -> csr_matrix:
        """For CSS codes only. Returns the part of the syndrome that
        corresponds to Z stabilizers.

        Parameters
        ----------
        syndrome: csr_matrix
            Syndrome as a sparse row of dimension 1xm, where m is the number
            of stabilizers.

        Returns
        -------
        z_syndrome: csr_matrix
            Syndrome reduced to X stabilizers
        """

        return syndrome[self.z_indices]

    def to_bsf(self, operator: Operator) -> csr_matrix:
        """Convert an operator (given as a dictionary qubit_location -> pauli)
        to a sparse row in the binary symplectic format.

        Parameters
        ----------
        operator: Dict[Tuple, str]
            Operator given as a dictionary that assigns a Pauli operator
            ('X', 'Y' or 'Z') to each qubit location in its support

        Returns
        -------
        bsf_operator: scipy.sparse.csr_matrix
            Sparse row of dimension 1x2n in the binary symplectic format
            (where n is the number of qubits)
        """
        bsf_operator = bsparse.zero_row(2*self.n)

        for qubit_location in operator.keys():
            if operator[qubit_location] in ['X', 'Y']:
                bsparse.insert_mod2(self.qubit_index[qubit_location], bsf_operator)
            if operator[qubit_location] in ['Y', 'Z']:
                bsparse.insert_mod2(self.n + self.qubit_index[qubit_location], bsf_operator)

        return bsf_operator

    def from_bsf(self, bsf_operator: csr_matrix) -> Operator:
        """Convert an operator given as a sparse row in the binary
        symplectic format to a dictionary qubit_location -> pauli.

        Parameters
        ----------
        bsf_operator: scipy.sparse.csr_matrix
            Sparse row of dimension 1x2n in the binary symplectic format
            (where n is the number of qubits)
        Returns
        -------
        operator: Dict[Tuple, str]
            Operator given as a dictionary that assigns a Pauli operator
            ('X', 'Y' or 'Z') to each qubit location in its support

        """
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

    def measure_syndrome(self, error: csr_matrix) -> csr_matrix:
        """Noiseless syndrome corresponding to a given Pauli error.

        Parameters
        ----------
        error: scipy.sparse.csr_matrix
            Error given as sparse row of dimension 1x2n in the binary
            symplectic format.

        Returns
        -------
        syndrome: scipy.sparse.csr_matrix
            Syndrome, as a sparse row of dimension 1xm (where m is the number
            of stabilizers)
        """

        return bcommute(self.stabilizer_matrix, error)

    def is_stabilizer(self, location: Tuple, stab_type: str = None):
        """Returns whether a given location in the coordinate system
        corresponds to a stabilizer or not
        """
        _is_stabilizer = (location in self.stabilizer_index) and\
                         (stab_type is None or self.stabilizer_type(location) == stab_type)

        return _is_stabilizer

    def is_qubit(self, location: Tuple):
        """Returns whether a given location in the coordinate system
        corresponds to a qubit or not. It is done by checking that the input
        location is a key in the dictionary `self.qubit_index`.

        Parameters
        ----------
        location : Tuple
            Location as a tuple of coordinates

        Returns
        -------
        Bool
            Whether the location is a qubit in the coordinate system.
        """
        return location in self.qubit_index

    @abstractmethod
    def get_qubit_coordinates(self) -> List[Tuple]:
        """Give the list of all the qubit coordinates, in a coordinate system
        that should contain both the qubits and the stabilizers.
        This function is used to set the attributes `self.qubit_coordinates`
        and `self.qubit_index`.

        Returns
        -------
        qubit_coordinates: List[Tuple]
            List of coordinates
        """

    @abstractmethod
    def get_stabilizer_coordinates(self) -> List[Tuple]:
        """Create list of stabilizer coordinates, in a coordinate system
        that should contain both the qubits and the stabilizers.
        This function is used to set the attributes `self.stabilizer_coordinates`
        and `self.stabilizer_index`.
        """

    @abstractmethod
    def qubit_axis(self, location: Tuple) -> str:
        """ Return the orientation of a qubit sitting at given location
        (as a string representing the axis 'x', 'y' or 'z').
        Useful when qubits have an orientation in space, for instance when
        they are edges, to help establish the visual representation of the
        code in the GUI, to simplify the construction of stabilizers,
        and to create Clifford deformations.

        Parameters
        ----------
        location: Tuple
            Location of the qubit in the coordinate system.

        Returns
        -------
        axis: str
            Either 'x', 'y' or 'z', depending on the orientation axis of the
            qubit.
        """

    @abstractmethod
    def stabilizer_type(self, location: Tuple) -> str:
        """ Return the type of a stabilizer sitting at a given location.
        E.g. 'vertex' or 'face' in toric codes
        """

    @abstractmethod
    def get_stabilizer(self, location: Tuple, deformed_axis: str = None) -> Operator:
        """ Returns a stabilizer, formatted as dictionary that assigns a Pauli
        operator ('X', 'Y' or 'Z') to each qubit location in the support of
        the stabilizer.

        For example, for a vertex stabilizer in the 2D toric code, we could have
        `get_stabilizer((1,1)) -> {(1,0): 'X', (0, 1): 'X', (2, 1): 'X', (1, 2): 'X'}`

        Parameters
        ----------
        location: Tuple
            Location of the stabilizer in the coordinate system
        deformed_axis: str, optional
            If given, represents an axis ('x', 'y' or 'z') that we want to
            Clifford-deform, by applying a Clifford transformation to all the
            qubits oriented along the given axis
            (e.g. `deformed_axis='x'` in the 2D toric code could give an
            XZZX surface code, where the transformation Pauli X <-> Z
            has been applied to all the vertical qubits of the code)

        Returns
        -------
        stabilizer: Dict[Tuple, str]
            Dictionary that assigns a Pauli operator ('X', 'Y' or 'Z') to each
            qubit location in the support of the stabilizer
        """

    @abstractmethod
    def get_logicals_x(self) -> List[Operator]:
        """Returns the list of logical X operators, where each operator is a
        dictionary that assigns a Pauli operator ('X', 'Y' or 'Z') to each
        qubit location in its support.

        Returns
        -------
        logicals: List[Dict[Tuple, str]]
            List of dictionaries, where each dictionary assign a Pauli
            operator ('X', 'Y' or 'Z') to each qubit location in the support
            of the logical operator.
        """

    @abstractmethod
    def get_logicals_z(self) -> List[Operator]:
        """Returns the list of logical Z operators, where each operator is a
        dictionary that assigns a Pauli operator ('X', 'Y' or 'Z') to each
        qubit location in its support.

        Returns
        -------
        logicals: List[Dict[Tuple, str]]
            List of dictionaries, where each dictionary assign a Pauli
            operator ('X', 'Y' or 'Z') to each qubit location in the support
            of the logical operator.
        """

    def stabilizer_representation(self,
                                  location: Tuple,
                                  rotated_picture=False,
                                  json_file=None) -> Dict:
        """Returns a dictionary of visualization parameters for the input
        stabilizer, that can be used by the web visualizer.

        It should contain 4 keys:
        - 'type': the type of stabilizer, e.g. 'vertex'
        - 'location': [x, y, z],
        - 'object': the type of object to use for visualization, e.g. 'sphere'
        - 'params': a dictionary of parameters for the chosen object

        Parameters
        ----------
        location: Tuple
            Coordinates of the stabilizer
        rotated_picture: bool
            For codes that have a rotated picture, can be used to differentiate
            the two types visualizations
        json_file: str
            File with the initial configuration for the code

        Returns
        -------
        representation: Dict
            Dictionary to send to the GUI
        """
        if json_file is None:
            json_file = os.path.join(os.environ['PANQEC_ROOT_DIR'], 'codes', 'gui-config.json')

        stab_type = self.stabilizer_type(location)

        with open(json_file, 'r') as f:
            data = json.load(f)

        code_name = self.id
        picture = 'rotated' if rotated_picture else 'kitaev'

        representation = data[code_name]['stabilizers'][picture][stab_type]
        representation['type'] = stab_type
        representation['location'] = location

        for activation in ['activated', 'deactivated']:
            color_name = representation['color'][activation]
            representation['color'][activation] = self.colormap[color_name]

        return representation

    def qubit_representation(self,
                             location: Tuple,
                             rotated_picture=False,
                             json_file=None) -> Dict:
        """Returns a dictionary of visualization parameters for the input
        qubit,  that can be used by the web visualizer.
        - 'location': [x, y, z],
        - 'object': the type of object to use for visualization, e.g. 'sphere'
        - 'params': a dictionary of parameters for the chosen object

        Parameters
        ----------
        location: Tuple
            Coordinates of the qubit
        rotated_picture: bool
            For codes that have a rotated picture, can be used to differentiate
            the two types visualizations
        json_file: str
            File with the initial configuration for the code

        Returns
        -------
        representation: Dict
            Dictionary to send to the GUI
        """
        if json_file is None:
            json_file = os.path.join(os.environ['PANQEC_ROOT_DIR'], 'codes', 'gui-config.json')

        with open(json_file, 'r') as f:
            data = json.load(f)

        code_name = self.id

        # if self.id == 'MyToric3DCode':
        #     print(data)
        #     print()
        #     print()
        #     print(data[code_name])
        #     print()
        #     print(data[code_name]['qubits'])
        #     print(data[code_name]['qubits'][picture])

        picture = 'rotated' if rotated_picture else 'kitaev'

        representation = data[code_name]['qubits'][picture]

        representation['params']['axis'] = self.qubit_axis(location)
        representation['location'] = location

        for pauli in ['I', 'X', 'Y', 'Z']:
            color_name = representation['color'][pauli]
            representation['color'][pauli] = self.colormap[color_name]

        return representation
