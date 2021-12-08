import itertools
from typing import Tuple, Optional, Dict
import numpy as np
from qecsim.model import StabilizerCode
from ._rhombic_pauli import RhombicPauli
from ..bpauli import bcommute


class RhombicCode(StabilizerCode):

    _size: Tuple[int, int, int]
    _shape: Tuple[int, int, int, int]
    _qubit_index: Dict[Tuple[int, int, int, int], int]
    _triangle_index: Dict[Tuple[int, int, int, int], int]
    _cube_index: Dict[Tuple[int, int, int], int]
    X_AXIS: int = 0
    Y_AXIS: int = 1
    Z_AXIS: int = 2
    _stabilizers = np.array([])
    _Hz = np.array([])
    _Hx = np.array([])
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
        self._shape = (3, L_x, L_y, L_z)
        self._size = (L_x, L_y, L_z)

        self._qubit_index = self._create_qubit_indices()
        self._triangle_index = self._create_triangle_indices()
        self._cube_index = self._create_cube_indices()

    # StabilizerCode interface methods.

    @property
    def n_k_d(self) -> Tuple[int, int, int]:
        return (np.product(self.shape), 3, min(self.size))

    @property
    def qubit_index(self) -> Dict[Tuple[int, int, int, int], int]:
        return self._qubit_index

    @property
    def triangle_index(self) -> Dict[Tuple[int, int, int, int], int]:
        return self._triangle_index

    @property
    def cube_index(self) -> Dict[Tuple[int, int, int], int]:
        return self._cube_index

    @property
    def n_cubes(self) -> int:
        return len(self.cube_index)

    @property
    def n_triangles(self) -> int:
        return len(self.triangle_index)

    @property
    def label(self) -> str:
        return 'Rhombic {}x{}x{}'.format(*self.size)

    @property
    def stabilizers(self) -> np.ndarray:
        if self._stabilizers.size == 0:
            cube_stabilizers = self.get_cube_X_stabilizers()
            triangle_stabilizers = self.get_triangle_Z_stabilizers()
            self._stabilizers = np.concatenate([
                cube_stabilizers,
                triangle_stabilizers,
            ])
        return self._stabilizers

    @property
    def Hz(self) -> np.ndarray:
        if self._Hz.size == 0:
            self._Hz = self.stabilizers[:self.n_cubes, :self.n_k_d[0]]
        return self._Hz

    @property
    def Hx(self) -> np.ndarray:
        if self._Hx.size == 0:
            self._Hx = self.stabilizers[self.n_cubes:, self.n_k_d[0]:]
        return self._Hx

    @property
    def logical_xs(self) -> np.ndarray:
        """The 3 logical X operators."""

        if self._logical_xs.size == 0:
            L_x, L_y, L_z = self.size
            logicals = []

            # Sheet of X operators normal to the z direction
            logical = RhombicPauli(self)
            for x in range(L_x):
                for y in range(L_y):
                    logical.site('X', (self.X_AXIS, x, y, 0))
                    logical.site('X', (self.Y_AXIS, x, y, 0))
            logicals.append(logical.to_bsf())

            # Sheet of X operators normal to the y direction
            logical = RhombicPauli(self)
            for x in range(L_x):
                for z in range(L_z):
                    logical.site('X', (self.X_AXIS, x, 0, z))
                    logical.site('X', (self.Z_AXIS, x, 0, z))
            logicals.append(logical.to_bsf())

            # Sheet of X operators normal to the x direction
            logical = RhombicPauli(self)
            for y in range(L_y):
                for z in range(L_z):
                    logical.site('X', (self.Y_AXIS, 0, y, z))
                    logical.site('X', (self.Z_AXIS, 0, y, z))
            logicals.append(logical.to_bsf())

            self._logical_xs = np.array(logicals, dtype=np.uint)

        return self._logical_xs

    @property
    def logical_zs(self) -> np.ndarray:
        """Get the 3 logical Z operators."""
        if self._logical_zs.size == 0:
            L_x, L_y, L_z = self.size
            logicals = []

            # Line of parallel Z operators along the x direction
            logical = RhombicPauli(self)
            for x in range(L_x):
                logical.site('Z', (self.Y_AXIS, x, 0, 0))
            logicals.append(logical.to_bsf())

            # Line of parallel Z operators along the y direction
            logical = RhombicPauli(self)
            for y in range(L_y):
                logical.site('Z', (self.Z_AXIS, 0, y, 0))
            logicals.append(logical.to_bsf())

            # Line of parallel Z operators along the z direction
            logical = RhombicPauli(self)
            for z in range(L_z):
                logical.site('Z', (self.X_AXIS, 0, 0, z))
            logicals.append(logical.to_bsf())

            self._logical_zs = np.array(logicals, dtype=np.uint)

        return self._logical_zs

    @property
    def size(self) -> Tuple[int, int, int]:
        """Dimensions of lattice."""
        return self._shape[1:]

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        """Shape of lattice for each qubit."""
        return self._shape

    def _create_qubit_indices(self):
        ranges = [range(length) for length in self.shape]
        coordinates = [
            (axis, x, y, z) for axis, x, y, z in itertools.product(*ranges)
        ]

        coord_to_index = {coord: i for i, coord in enumerate(coordinates)}

        return coord_to_index

    def _create_triangle_indices(self):
        ranges = [range(length) for length in (4,) + self.size]
        coordinates = [
            (axis, x, y, z) for axis, x, y, z in itertools.product(*ranges)
        ]

        coord_to_index = {coord: i for i, coord in enumerate(coordinates)}

        return coord_to_index

    def _create_cube_indices(self):
        ranges = [range(length) for length in self.size]
        coordinates = []
        for x, y, z in itertools.product(*ranges):
            if (x + y + z) % 2 == 1:
                coordinates.append((x, y, z))

        coord_to_index = {coord: i for i, coord in enumerate(coordinates)}

        return coord_to_index

    def get_triangle_Z_stabilizers(self) -> np.ndarray:
        triangle_stabilizers = []
        ranges = [range(length) for length in (4,) + self.size]

        # Z operators for each vertex for each position.
        for axis, L_x, L_y, L_z in itertools.product(*ranges):
            operator = RhombicPauli(self)
            operator.triangle('Z', axis, (L_x, L_y, L_z))
            triangle_stabilizers.append(operator.to_bsf())
        return np.array(triangle_stabilizers, dtype=np.uint)

    def get_cube_X_stabilizers(self) -> np.ndarray:
        cube_stabilizers = []
        ranges = [range(length) for length in self.size]

        for L_x, L_y, L_z in itertools.product(*ranges):
            if (L_x + L_y + L_z) % 2 == 1:
                operator = RhombicPauli(self)
                operator.cube('X', (L_x, L_y, L_z))
                cube_stabilizers.append(operator.to_bsf())
        return np.array(cube_stabilizers, dtype=np.uint)

    def measure_syndrome(self, error: RhombicPauli) -> np.ndarray:
        """Perfectly measure syndromes given Pauli error."""
        return bcommute(self.stabilizers, error.to_bsf())
