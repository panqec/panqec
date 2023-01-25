import numpy as np
from panqec.codes import StabilizerCode
from scipy.sparse import dok_matrix
from typing import Dict, Tuple, List
import networkx as nx
import stim
import sinter
import plotly
import plotly.graph_objs as go


def add_detector(circuit, qubits):
    n_qubits = circuit.num_qubits
    circuit.append("DETECTOR", [stim.target_rec(i - n_qubits) for i in qubits])


def add_observable(circuit, qubits):
    n_qubits = circuit.num_qubits
    circuit.append("OBSERVABLE_INCLUDE",
                   [stim.target_rec(i - n_qubits) for i in qubits],
                   0)


class FoliatedCode:
    def __init__(self, base_code: StabilizerCode, n_layers: int):
        self.base_code = base_code
        self.n_layers = n_layers

        self._qubit_index = {}
        self._stabilizer_index = {}
        self._qubit_coordinates = []
        self._ancilla_coordinates = []
        self._primal_coordinates = []
        self._dual_coordinates = []
        self._stabilizer_coordinates = []

        self._H_primal = None
        self._H_dual = None
        self._cs_graph = None

    @property
    def params(self) -> Dict:
        return {
            'base_code': self.base_code.id,
            'base_code_params': self.base_code.params,
            'n_layers': self.n_layers
        }

    @property
    def id(self) -> str:
        return self.__class__.__name__

    @property
    def n(self) -> int:
        return len(self.qubit_coordinates)

    @property
    def k(self) -> int:
        return 1  # temporary
        # return self.base_code.k

    @property
    def d(self) -> int:
        return self.base_code.d

    @property
    def n_primal(self) -> int:
        return len(self.primal_coordinates)

    @property
    def n_dual(self) -> int:
        return len(self.dual_coordinates)

    @property
    def n_stabilizers(self) -> int:
        return len(self.stabilizer_coordinates)

    def is_qubit(self, coord: List[Tuple]) -> bool:
        return (coord in self.qubit_coordinates)

    def is_ancilla(self, coord: List[Tuple]) -> bool:
        return (coord in self.ancilla_coordinates)

    def is_data(self, coord: List[Tuple]) -> bool:
        return (not self.is_ancilla(coord))

    def is_primal(self, coord: List[Tuple]) -> bool:
        return ((self.is_ancilla(coord) and coord[-1] % 2 == 0) or
                (self.is_data(coord) and coord[-1] % 2 == 1))

    def is_dual(self, coord: List[Tuple]) -> bool:
        return not self.is_primal(coord)

    @property
    def primal_coordinates(self) -> List[Tuple]:
        if len(self._primal_coordinates) == 0:
            self._primal_coordinates = [
                coord for coord in self.qubit_coordinates
                if self.is_primal(coord)
            ]
        return self._primal_coordinates

    @property
    def dual_coordinates(self) -> List[Tuple]:
        if len(self._dual_coordinates) == 0:
            self._dual_coordinates = [
                coord for coord in self.qubit_coordinates
                if self.is_dual(coord)
            ]
        return self._dual_coordinates

    @property
    def qubit_coordinates(self) -> List[Tuple]:
        if len(self._qubit_coordinates) == 0:
            for coord in self.base_code.qubit_coordinates:
                for i in range(self.n_layers):
                    self._qubit_coordinates.append((*coord, i))

            self._qubit_coordinates += self.ancilla_coordinates

        return self._qubit_coordinates

    @property
    def ancilla_coordinates(self) -> List[Tuple]:
        if len(self._ancilla_coordinates) == 0:
            for coord in self.base_code.gauge_coordinates:
                for i in range(self.n_layers):
                    paulis = set(
                        self.base_code.get_gauge_operator(coord).values()
                    )
                    if ((i % 2 == 0 and paulis == {'X'})
                            or (i % 2 == 1 and paulis == {'Z'})):
                        self._ancilla_coordinates.append((*coord, i))

        return self._ancilla_coordinates

    @property
    def stabilizer_coordinates(self) -> List[Tuple]:
        if len(self._stabilizer_coordinates) == 0:
            for coord in self.base_code.stabilizer_coordinates:
                for i in range(self.n_layers):
                    paulis = set(
                        self.base_code.get_stabilizer(coord).values()
                    )
                    if ((i % 2 == 0 and paulis == {'Z'})
                            or (i % 2 == 1 and paulis == {'X'})):
                        self._stabilizer_coordinates.append((*coord, i))

        return self._stabilizer_coordinates

    def get_stabilizer(self, coord: Tuple):
        stab = {}
        base_stab = self.base_code.get_stabilizer(coord[:-1])

        for pauli_coord in base_stab:
            stab[(*pauli_coord, coord[-1])] = 'X'

        gauge_ops = self.base_code.get_stabilizer_gauge_operators(coord[:-1])

        ancillas = []
        for gauge_coord in gauge_ops:
            ancillas.append((*gauge_coord, coord[-1]+1))
            ancillas.append((*gauge_coord, coord[-1]-1))

        for qubit in ancillas:
            if self.is_qubit(qubit):
                stab[qubit] = 'X'

        return stab

    @property
    def qubit_index(self) -> Dict[Tuple, int]:
        """Dictionary that assigns an index to a given
        qubit coordinate"""

        if len(self._qubit_index) == 0:
            self._qubit_index = {
                coord: i for i, coord in enumerate(self.qubit_coordinates)
            }

        return self._qubit_index

    @property
    def stabilizer_index(self) -> Dict[Tuple, int]:
        """Dictionary that assigns an index to a given
        stabilizer coordinate"""

        if len(self._stabilizer_index) == 0:
            self._stabilizer_index = {
                coord: i for i, coord in enumerate(self.stabilizer_coordinates)
            }

        return self._stabilizer_index

    @property
    def H_primal(self):
        if self._H_primal is None:
            sparse_dict: Dict = dict()

            self._H_primal = dok_matrix(
                (self.n_stabilizers, self.n_primal),
                dtype='uint8'
            )

            for i_stab, stabilizer_coord in enumerate(
                self.stabilizer_coordinates
            ):
                stabilizer_op = self.get_stabilizer(stabilizer_coord)

                for qubit_coord in stabilizer_op.keys():
                    if self.is_primal(qubit_coord):
                        i_qubit = self.qubit_index[qubit_coord]
                        sparse_dict[(i_stab, i_qubit)] = 1

            self._H_primal._update(sparse_dict)
            self._H_primal = self._H_primal.tocsr()

        return self._H_primal

    @property
    def H_dual(self):
        if self._H_dual is None:
            sparse_dict: Dict = dict()

            self._H_dual = dok_matrix(
                (self.n_stabilizers, self.n_dual),
                dtype='uint8'
            )

            for i_stab, stabilizer_coord in enumerate(
                self.stabilizer_coordinates
            ):
                stabilizer_op = self.get_stabilizer(stabilizer_coord)

                for qubit_coord in stabilizer_op.keys():
                    if self.is_dual(qubit_coord):
                        i_qubit = self.qubit_index[qubit_coord]
                        sparse_dict[(i_stab, i_qubit)] = 1

            self._H_dual._update(sparse_dict)
            self._H_dual = self._H_dual.tocsr()

        return self._H_dual

    @property
    def cs_graph(self):
        if self._cs_graph is None:
            self._cs_graph = nx.Graph()

            for qubit_idx, qubit_coord in enumerate(self.qubit_coordinates):
                self._cs_graph.add_node(qubit_idx, pos=qubit_coord)

                # Horizontal connections
                if self.is_ancilla(qubit_coord):
                    base_stab = self.base_code.get_gauge_operator(
                        qubit_coord[:-1]
                    )
                    for pauli_coord in base_stab.keys():
                        new_idx = self.qubit_index[
                            (*pauli_coord, qubit_coord[-1])
                        ]

                        self._cs_graph.add_edge(qubit_idx, new_idx)

                # Vertical connections
                else:
                    if qubit_coord[-1] < self.n_layers - 1:
                        new_idx = self.qubit_index[
                            (*qubit_coord[:-1], qubit_coord[-1]+1)
                        ]

                        self._cs_graph.add_edge(qubit_idx, new_idx)

        return self._cs_graph

    def qubit_representation(self, coordinates):
        if self.is_ancilla(coordinates):
            rep = self.base_code.gauge_representation(coordinates[:-1])
        else:
            rep = self.base_code.qubit_representation(coordinates[:-1])

        return {'location': (*rep['location'], coordinates[-1])}

    def get_logicals(self):
        """List of list of coordinates for all the logicals"""
        all_logicals = []

        # if self.n_layers % 2 == 1:
        for logical in self.base_code.get_logicals_z():
            all_logicals.append([])
            for qubit_coord in logical.keys():
                for i in range(0, self.n_layers, 2):
                    all_logicals[-1].append((*qubit_coord, i))
        # else:
        #     for logical in self.base_code.get_logicals_x():
        #         all_logicals.append([])
        #         for qubit_coord in logical.keys():
        #             for i in range(1, self.n_layers, 2):
        #                 all_logicals[-1].append((*qubit_coord, i))

        return all_logicals

    def draw_cs_graph(self, primal_color=None, dual_color=None):
        if primal_color is None:
            primal_color = plotly.colors.DEFAULT_PLOTLY_COLORS[3]
        if dual_color is None:
            dual_color = plotly.colors.DEFAULT_PLOTLY_COLORS[0]

        color_map = [primal_color if self.is_primal(q) else dual_color
                     for q in self.qubit_coordinates]

        x_nodes = list(map(
            lambda coord: self.qubit_representation(coord)['location'][0],
            self.qubit_coordinates
        ))
        y_nodes = list(map(
            lambda coord: self.qubit_representation(coord)['location'][1],
            self.qubit_coordinates
        ))
        z_nodes = list(map(
            lambda coord: self.qubit_representation(coord)['location'][2],
            self.qubit_coordinates
        ))

        def edge_location(e):
            return (
                self.qubit_representation(
                    self.qubit_coordinates[e[0]]
                )['location'],
                self.qubit_representation(
                    self.qubit_coordinates[e[1]]
                )['location']
            )

        x_edges = list(map(
            lambda e: [edge_location(e)[0][0], edge_location(e)[1][0], None],
            self.cs_graph.edges
        ))
        y_edges = list(map(
            lambda e: [edge_location(e)[0][1], edge_location(e)[1][1], None],
            self.cs_graph.edges
        ))
        z_edges = list(map(
            lambda e: [edge_location(e)[0][2], edge_location(e)[1][2], None],
            self.cs_graph.edges
        ))

        trace_nodes = go.Scatter3d(
            x=x_nodes,
            y=y_nodes,
            z=z_nodes,
            mode='markers',
            marker={
                'size': 5,
                'opacity': 0.8,
                'color': color_map
            }
        )

        trace_edges = go.Scatter3d(
            x=np.ravel(x_edges),
            y=np.ravel(y_edges),
            z=np.ravel(z_edges),
            mode='lines',
            line=dict(color='rgb(0,0,0)', width=2)
        )

        # Configure the layout.
        layout = go.Layout(
            margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
            scene=dict(
                xaxis={'dtick': 1},
                yaxis={'dtick': 1},
                zaxis={'dtick': 1}
            ),
            legend={'x': 0}
        )

        data = [trace_nodes, trace_edges]

        plot_figure = go.Figure(data=data, layout=layout)

        plotly.offline.iplot(plot_figure)

    def get_circuit(self, error_rate=0.):
        all_qubits = list(range(self.n))

        circuit = stim.Circuit()
        circuit.append("H", all_qubits)

        for e in self.cs_graph.edges:
            circuit.append("CZ", [e[0], e[1]])

        circuit.append("Z_ERROR", all_qubits, error_rate)

        circuit.append("MX", all_qubits)

        for logical in self.get_logicals():
            qubit_indices = list(map(lambda a: self.qubit_index[a], logical))
            # print("Logical supported on ", qubit_indices)

            add_observable(circuit, qubit_indices)

        for stab_coord in self.stabilizer_coordinates:
            qubit_coords = self.get_stabilizer(stab_coord).keys()
            # print(qubit_coords)
            qubit_indices = map(lambda a: self.qubit_index[a], qubit_coords)
            add_detector(circuit, qubit_indices)

        return circuit

    def run(self, n_runs, error_rate):
        circuit = self.get_circuit(error_rate)

        sampler = circuit.compile_detector_sampler()

        detection_events, observable_flips = sampler.sample(
            n_runs, separate_observables=True
        )
        detector_error_model = circuit.detector_error_model(
            decompose_errors=True
        )

        predictions = sinter.predict_observables(
            dem=detector_error_model,
            dets=detection_events,
            decoder='pymatching',
        )

        effective_error = []
        success = []
        codespace = []
        for actual_flip, predicted_flip in zip(observable_flips, predictions):
            if not np.array_equal(actual_flip, predicted_flip):
                effective_error.append([1])
                success.append(0)
            else:
                effective_error.append([0])
                success.append(1)

            codespace.append(1)  # Temporary

        return {
            'effective_error': effective_error,
            'success': success,
            'codespace': codespace
        }

    def get_logical_error_rate(self, n_runs, error_rate):
        results = self.run(n_runs, error_rate)

        logical_error_rate = 1 - np.sum(results['success']) / n_runs

        return logical_error_rate
