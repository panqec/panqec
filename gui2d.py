import numpy as np
import time
from qecsim.models.toric import ToricCode
from ursina import *

from bn3d.bp_os_decoder import BeliefPropagationOSDDecoder

config = {'sizes': dict(), 'colors': dict()}

config['sizes']['length_edge'] = 1
config['sizes']['width_edge'] = config['sizes']['length_edge'] / 10
config['sizes']['vertex'] = 0.2
config['sizes']['vertex_activated'] = config['sizes']['vertex'] * 1.5

config['colors']['background'] = color.rgb(101, 29, 49)
config['colors']['error'] = color.rgb(255, 61, 61)
config['colors']['highlight'] = color.orange
config['colors']['edge'] = color.rgb(238, 238, 238)
config['colors']['vertex'] = config['colors']['edge']
config['colors']['vertex_activated'] = color.rgb(241, 194, 50)


class Grid:
    def __init__(self, code, boundary):
        self.qubit_indices = np.array(code._indices)
        self.n_rows, self.n_cols = code.size
        self.n_qubits = code.n_k_d[0]
        self.n_stabilizers = self.n_rows * self.n_cols

        self.qubits = np.zeros(self.n_qubits, dtype=Qubit)
        self.stabilizers = np.zeros(self.n_stabilizers, dtype=Qubit)

        self.app = Ursina()

        window.fps_counter.enabled = False
        window.title = 'Surface code'
        window.borderless = False
        window.exit_button.visible = False
        window.color = config['colors']['background']

        L = config['sizes']['length_edge']
        self.init_x = -(L*self.n_cols + L/2) / 2
        self.init_y = -(L*self.n_rows + L/2) / 2

        self.qubit_grid = {'h': {}, 'v': {}}
        self.stab_grid = {}

        # Construct qubits, stabilizers and the corresponding graphical entities
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                # Stabilizer
                self.construct_stabilizer(row, col)

                # Horizontal qubits
                if col < self.n_cols - 1:
                    self.construct_qubit(row, col, 'h', periodic=False)

                # Vertical qubits
                if row < self.n_rows - 1:
                    self.construct_qubit(row, col, 'v', periodic=False)

        # Boundary conditions
        if boundary['h'] != 'rough':
            periodic = (boundary['h'] == 'periodic')
            for row in range(self.n_rows):
                # Left
                qubit_left = self.construct_qubit(row, -1, 'h', periodic)

                # Right
                qubit = qubit_left if periodic else None
                self.construct_qubit(row, self.n_cols-1, 'h', periodic, qubit=qubit)

        if boundary['v'] != 'rough':
            periodic = (boundary['v'] == 'periodic')
            for col in range(self.n_cols):
                # Bottom
                qubit_bottom = self.construct_qubit(-1, col, 'v', periodic)

                # Top
                qubit = qubit_bottom if periodic else None
                self.construct_qubit(self.n_rows-1, col, 'v', periodic, qubit=qubit)

        # Construct adjacency matrix
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                qubits = []
                if (row, col) in self.qubit_grid['h'].keys():
                    qubits.append(self.qubit_grid['h'][(row, col)])
                if (row, col) in self.qubit_grid['v'].keys():
                    qubits.append(self.qubit_grid['v'][(row, col)])
                if (row, col-1) in self.qubit_grid['h'].keys():
                    qubits.append(self.qubit_grid['h'][(row, col-1)])
                if (row-1, col) in self.qubit_grid['v'].keys():
                    qubits.append(self.qubit_grid['v'][(row-1, col)])

                self.stab_grid[row, col].adj_qubits = qubits

                for q in qubits:
                    q.adj_stabilizers.append(self.stab_grid[row, col]) 

    def run(self):
        self.app.run()
    
    def construct_stabilizer(self, row, col):
        code_row = (self.n_rows - row - 2) % self.n_rows 
        code_col = col % self.n_cols

        stab_id = code_row * self.n_rows + code_col

        L = config['sizes']['length_edge']
        size_vertex = config['sizes']['vertex']
        color_vertex = config['colors']['vertex']

        pos = (self.init_x + L*col + L/2, self.init_y + L*(row+1))
        circle = Entity(model='circle', color=color_vertex, scale=(size_vertex, size_vertex), position=pos, always_on_top=True)

        stab = Stabilizer(stab_id, circle)
        self.stabilizers[stab_id] = stab
        self.stab_grid[(row, col)] = stab

        # text = Text(f"{(code_row, code_col)}", color=color.red, position=(0.12*col-0.3, 0.12*row-0.2))
        # text = Text(f"{stab_id}", color=color.red, position=(0.14*pos[0], 0.14*pos[1]))

    def construct_qubit(self, row, col, orientation, periodic, qubit=None):
        orientation_id = 0 if orientation == 'h' else 1

        # Rows and cols in the qecsim format. Need to manage row=-1 (for periodic edges)
        code_row = self.n_rows - 1 - row % self.n_rows if orientation == 'h' else (self.n_rows - row - 2) % n_rows
        code_col = col % self.n_cols

        qubit_id = np.where(np.all(self.qubit_indices == (orientation_id, code_row, code_col), axis=1))[0][0]

        # print((orientation_id, row, col), qubit_id)

        if qubit is None:
            qubit = Qubit(qubit_id, row, col, orientation)

        init_pos = (self.init_x, self.init_y)
        edge = Edge(qubit, row, col, init_pos, periodic=periodic)
        qubit.edges.append(edge)

        self.qubit_grid[orientation][(row, col)] = qubit
        self.qubits[qubit_id] = qubit

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
    def __init__(self, id, row, col, orientation):
        self.id = id
        self.has_error = False
        self.orientation = orientation
        self.edges = []
        self.row, self.col = row, col
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
    def __init__(self, qubit, row, col, init_pos, periodic=False):
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
    if held_keys['backspace']:
        print("Reset errors...\n")
        for qubit in grid.qubits:
            if qubit.has_error:
                qubit.insert_error()
        
        time.sleep(0.5)

    if held_keys['r']:
        print("Add random errors...\n")
        grid.add_random_errors(p)
        
        time.sleep(0.5)

    if held_keys['d']:
        print("Start decoding...\n")

        syndrome = grid.get_syndrome()
        syndrome = np.concatenate([np.zeros(len(syndrome), dtype=np.int64), syndrome])
        decoder = BeliefPropagationOSDDecoder()

        correction = decoder.decode(code, syndrome)

        for i_qubit, qubit in enumerate(correction[grid.n_qubits:]):
            if qubit == 1:
                grid.qubits[i_qubit].insert_error()
        
        time.sleep(0.5)


if __name__ == "__main__":
    n_rows = 7
    n_cols = 7
    max_iter_bp = 10
    p = 0.1

    boundary = {'v': 'periodic',
                'h': 'periodic'}

    code = ToricCode(n_rows, n_cols)
    grid = Grid(code, boundary)
    print("Number of qubits", len(grid.qubits))
    grid.run()

    # grid.add_random_errors(p)

    # neighbor_data = grid.get_neighbor_data()
    # neighbor_parity = grid.get_neighbor_parity()

    # syndrome = grid.get_syndrome()
    # errors = bp_osd_decoder(neighbor_data, neighbor_parity, syndrome, max_iter=max_iter_bp)
