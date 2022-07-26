from typing import Optional, List, Tuple, Dict
from ._stabilizer_code import StabilizerCode
from abc import ABCMeta, abstractmethod
from scipy.sparse import csr_matrix, dok_matrix
from panqec import bsparse

Operator = Dict[Tuple, str]  # Coordinate to pauli ('X', 'Y' or 'Z')


class SubsystemCode(StabilizerCode, metaclass=ABCMeta):

    _gauge_index: Dict[Tuple, int]
    _gauge_coordinates: List[Tuple]
    _gauge_matrix: csr_matrix

    def __init__(
        self, L_x: int,
        L_y: Optional[int] = None,
        L_z: Optional[int] = None,
        **kwargs
    ):
        """Constructor for the SubsystemCode class

        Parameters
        ----------
        L_x : int
            Dimension of the lattice in the x direction (or in all directions
            if L_y and L_z are not given)
        L_y: int, optional
            Dimension of the lattice in the y direction
        L_z: int, optional
            Dimension of the lattice in the z direction
        """
        super().__init__(L_x, L_y=L_x, L_z=L_z, **kwargs)
        self._gauge_coordinates = []
        self._gauge_index = dict()
        self._gauge_matrix = bsparse.empty_row(2*self.n)

    @property
    def n_gauge(self) -> int:
        """Number of gauge generators."""
        return len(self.gauge_index)

    @property
    def gauge_index(self) -> Dict[Tuple, int]:
        """Dictionary that assigns an index to a given gauge location"""
        if len(self._gauge_index) == 0:
            self._gauge_index = {
                loc: i
                for i, loc in enumerate(self.gauge_coordinates)
            }

        return self._gauge_index

    @abstractmethod
    def get_gauge_coordinates(self) -> List[Tuple]:
        """Get the coordinates of the gauge operators.

        Returns
        -------
        coordinates : List[Tuple]
            List of coordinates associated with each gauge operator.
        """
        pass

    @abstractmethod
    def get_gauge(self, location: Tuple) -> Operator:
        """Get the gauge operator at a given location.

        Parameters
        ----------
        location : Tuple
            Location of the gauge operator.

        Returns
        -------
        operator : Dict[Tuple, str]
            Gauge operator defined at the given location.
        """
        pass

    @abstractmethod
    def get_stabilizers_from_gauges(self) -> Dict[Tuple, List[Tuple]]:
        """Products of gauges that give each stabilizer.

        Returns
        -------
        mapping : Dict[int, List[int]]
            For each stabilizer generator index, a list of gauge generator
            indices whose product produces the stabilizer generator.
        """
        pass

    @property
    def gauge_coordinates(self) -> List[Tuple]:
        """List of all the coordinates that contain a gauge operator."""

        if len(self._gauge_coordinates) == 0:
            self._gauge_coordinates = self.get_gauge_coordinates()

        return self._gauge_coordinates

    @property
    def gauge_matrix(self) -> csr_matrix:
        """Binary matrix with rows representing gauge generators."""
        if bsparse.is_empty(self._gauge_matrix):
            sparse_dict: Dict = dict()
            self._gauge_matrix = dok_matrix(
                (self.n_gauge, 2*self.n),
                dtype='uint8'
            )

            for i_stab, gauge_location in enumerate(self.gauge_coordinates):
                gauge_op = self.get_gauge(gauge_location)

                for qubit_location in gauge_op.keys():
                    if gauge_op[qubit_location] in ['X', 'Y']:
                        i_qubit = self.qubit_index[qubit_location]
                        if (i_stab, i_qubit) in sparse_dict.keys():
                            sparse_dict[(i_stab, i_qubit)] += 1
                        else:
                            sparse_dict[(i_stab, i_qubit)] = 1
                    if gauge_op[qubit_location] in ['Y', 'Z']:
                        i_qubit = self.n + self.qubit_index[qubit_location]
                        if (i_stab, i_qubit) in sparse_dict.keys():
                            sparse_dict[(i_stab, i_qubit)] += 1
                        else:
                            sparse_dict[(i_stab, i_qubit)] = 1

            self._gauge_matrix._update(sparse_dict)
            self._gauge_matrix = self._gauge_matrix.tocsr()
            self._gauge_matrix.data %= 2

        return self._gauge_matrix


    def get_stabilizer(
        self, location: Tuple, deformed_axis: str = None
    ) -> Operator:
        operator: Operator = dict()
        mapping = self.get_stabilizers_from_gauges()
        # TODO
        return operator
