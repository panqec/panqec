import numpy as np
import time
import itertools

from qecsim.models.toric import ToricCode
from bn3d.tc3d import ToricCode3D
from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController

from bn3d.bp_os_decoder_2d import BeliefPropagationOSDDecoder

config = {'sizes': dict(), 'colors': dict()}

config['sizes']['length_edge'] = 1
config['sizes']['width_edge'] = config['sizes']['length_edge'] / 10
config['sizes']['vertex'] = 1
config['sizes']['vertex_activated'] = config['sizes']['vertex'] * 1.5

config['colors']['background'] = color.rgb(101, 29, 49)
config['colors']['error'] = color.rgb(255, 61, 61)
config['colors']['highlight'] = color.orange
config['colors']['edge'] = color.rgb(238, 238, 238)
config['colors']['vertex'] = config['colors']['edge']
config['colors']['vertex_activated'] = color.rgb(241, 194, 50)


class Grid3D:
    def __init__(self, code, boundary):
        self.code = code
        self.n_qubits = code.n_k_d[0]
        self.size = np.array(code.shape[1:], dtype=np.uint)
        self.qubits = np.zeros(code.shape, dtype=Qubit)
        self.v_stabilizers = np.zeros(code.shape, dtype=Qubit)
        self.f_stabilizers = np.zeros(code.shape, dtype=Qubit)

        self.app = Ursina()

        ground = Entity(model='plane', color=color.gold, scale=(20, 1, 20))

        player = FirstPersonController()

        window.fps_counter.enabled = False
        window.title = '3D Toric code'
        window.borderless = False
        window.exit_button.visible = False
        window.color = config['colors']['background']

        L = config['sizes']['length_edge']
        self.origin = -(L * self.size + L / 2) / 2

        # Construct qubits, stabilizers and the corresponding graphical entities
        ranges = [range(length) for length in self.size]
        # for x, y, z in itertools.product(*ranges):
        #     # Vertex stabilizer
        #     self.construct_v_stabilizer(x, y, z)

            # Qubits
            # for axis in range(3):
            #     self.construct_qubit(axis, x, y, z)

        # Construct adjacency matrix
        # for row in range(self.n_rows):
        #     for col in range(self.n_cols):
        #         qubits = []
        #         if (row, col) in self.qubit_grid['h'].keys():
        #             qubits.append(self.qubit_grid['h'][(row, col)])
        #         if (row, col) in self.qubit_grid['v'].keys():
        #             qubits.append(self.qubit_grid['v'][(row, col)])
        #         if (row, col-1) in self.qubit_grid['h'].keys():
        #             qubits.append(self.qubit_grid['h'][(row, col-1)])
        #         if (row-1, col) in self.qubit_grid['v'].keys():
        #             qubits.append(self.qubit_grid['v'][(row-1, col)])
        #
        #         self.stab_grid[row, col].adj_qubits = qubits
        #
        #         for q in qubits:
        #             q.adj_stabilizers.append(self.stab_grid[row, col])

    def run(self):
        self.app.run()
    
    def construct_v_stabilizer(self, x, y, z):
        L = config['sizes']['length_edge']
        size_vertex = config['sizes']['vertex']
        color_vertex = config['colors']['vertex']

        sphere = Entity(model='sphere', color=color_vertex, scale=(size_vertex, size_vertex), position=[x, y, z],
                        always_on_top=True)

        stab = Stabilizer(sphere)
        self.v_stabilizers[x, y, z] = stab

        # text = Text(f"{(code_row, code_col)}", color=color.red, position=(0.12*col-0.3, 0.12*row-0.2))
        # text = Text(f"{stab_id}", color=color.red, position=(0.14*pos[0], 0.14*pos[1]))

    def construct_qubit(self, axis, x, y, z, qubit=None):
        if qubit is None:
            qubit = Qubit(axis, x, y, z)

        edge = Edge(qubit)
        qubit.edges.append(edge)

        return qubit

    def add_random_errors(self, p):
        for qubit in self.qubits:
            if np.random.rand() < p:
                qubit.insert_error()

    def get_neighbor_data(self):
        neighbor_data = []
        for qubit in self.qubits:
            neighbor_data.append([stab.id for stab in qubit.adj_stabilizers])
        return neighbor_data

    def get_neighbor_parity(self):
        neighbor_parity = []
        for stab in self.stabilizers:
            neighbor_parity.append([qubit.id for qubit in stab.adj_qubits])
        return neighbor_parity

    def get_errors(self):
        return np.array([1 if q.has_error else 0 for q in self.qubits] + [0 for _ in range(self.n_qubits)])

    def get_syndrome(self):
        return np.array([1 if stab.is_activated else 0 for stab in self.stabilizers], dtype=np.int64)


class Qubit:
    def __init__(self, axis, x, y, z):
        self.id = id
        self.has_error = False
        self.axis = axis
        self.x, self.y, self.z = x, y, z
        self.edges = []
        self.adj_stabilizers = []

    def insert_error(self):
        for edge in self.edges:
            if not self.has_error:
                edge.color = config['colors']['error']
            else:
                edge.color = config['colors']['edge']

        if not self.has_error:
            self.has_error = True
        else:
            self.has_error = False

        for stab in self.adj_stabilizers:
            n_errors = 0
            for q in stab.adj_qubits:
                if q.has_error:
                    n_errors += 1
                if n_errors % 2 == 1:
                    stab.activate()
                else:
                    stab.deactivate()


class Stabilizer:
    def __init__(self, id, circle=None):
        self.id = id
        self.is_activated = False
        self.circle = circle
        self.adj_qubits = []

    def activate(self):
        self.is_activated = True
        if self.circle is not None:
            color_vertex_activated = config['colors']['vertex_activated']
            size_vertex_activated = config['sizes']['vertex_activated']

            self.circle.color = color_vertex_activated
            self.circle.scale = (size_vertex_activated, size_vertex_activated)

    def deactivate(self):
        self.is_activated = False
        if self.circle is not None:
            color_vertex = config['colors']['vertex']
            size_vertex = config['sizes']['vertex']

            self.circle.color = color_vertex
            self.circle.scale = (size_vertex, size_vertex)


class Edge(Button):
    def __init__(self, qubit, origin, periodic=False):
        self.qubit = qubit
        self.periodic = periodic
        self.row = row
        self.col = col
        init_x, init_y = init_pos

        length_edge = config['sizes']['length_edge']
        width_edge = config['sizes']['width_edge']

        length = length_edge if not periodic else length_edge/2
        if qubit.orientation == "v":
            pos = [init_x + length_edge*col + length_edge/2, init_y + length_edge*row + 3*length_edge/2]
            scale = (width_edge, length)
            if periodic:
                if row == -1:
                    pos[1] += length_edge/4
                else:
                    pos[1] -= length_edge/4
        elif qubit.orientation == "h":
            pos = [init_x + length_edge*(col+1), init_y + length_edge*row + length_edge]
            scale = (length, width_edge)
            if periodic:
                if col == -1:
                    pos[0] += length_edge/4
                else:
                    pos[0] -= length_edge/4

        # text = Text(f"{self.qubit.id}", color=color.black, position=(0.14*pos[0], 0.14*pos[1]))

        super().__init__(
            parent = scene,
            position = pos,
            scale = scale,
            color = config['colors']['edge'],
            highlight_color = config['colors']['highlight']
        )

    def input(self, key):
        if self.hovered:
            if key == 'left mouse down':
                self.qubit.insert_error()


def update():
    pass
    # if held_keys['backspace']:
    #     print("Reset errors...\n")
    #     for qubit in grid.qubits:
    #         if qubit.has_error:
    #             qubit.insert_error()
    #
    #     time.sleep(0.5)
    #
    # if held_keys['r']:
    #     print("Add random errors...\n")
    #     grid.add_random_errors(p)
    #
    #     time.sleep(0.5)
    #
    # if held_keys['d']:
    #     print("Start decoding...\n")
    #
    #     syndrome = grid.get_syndrome()
    #     syndrome = np.concatenate([np.zeros(len(syndrome), dtype=np.int64), syndrome])
    #     decoder = BeliefPropagationOSDDecoder()
    #
    #     correction = decoder.decode(code, syndrome)
    #
    #     for i_qubit, qubit in enumerate(correction[grid.n_qubits:]):
    #         if qubit == 1:
    #             grid.qubits[i_qubit].insert_error()
        
        # time.sleep(0.5)


if __name__ == "__main__":
    size = (3, 3, 3)
    max_iter_bp = 10
    p = 0.1

    boundary = {'v': 'periodic',
                'h': 'periodic'}

    code = ToricCode3D(*size)
    # grid = Grid3D(code, boundary)
    # grid.run()

    app = Ursina()

    ground = Entity(model='plane', color=color.gold, scale=(20, 1, 20), collider="box")

    window.fps_counter.enabled = False
    window.title = '3D Toric code'
    window.borderless = False
    window.exit_button.visible = False
    window.color = config['colors']['background']

    player = FirstPersonController()
    player.gravity = False

    app.run()

    # grid.add_random_errors(p)

    # neighbor_data = grid.get_neighbor_data()
    # neighbor_parity = grid.get_neighbor_parity()

    # syndrome = grid.get_syndrome()
    # errors = bp_osd_decoder(neighbor_data, neighbor_parity, syndrome, max_iter=max_iter_bp)
