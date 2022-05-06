from typing import Dict, Any, Tuple, Optional, Iterator
from itertools import product
import numpy as np

from .model import SpinModel, DisorderModel, VectorObservable
from .observables import (
    Magnetization, Susceptibility0, Susceptibilitykmin, Energy
)


class LoopModel2D(SpinModel):
    """Loop Model in 2D."""

    label: str = ''
    parameters: Dict[str, Any] = {}
    spin_shape: Tuple[int, int] = (0, 0)
    bond_shape: Tuple[int, int] = (0, 0)
    disorder: np.ndarray = np.array([])
    spins: np.ndarray = np.array([])
    couplings: np.ndarray = np.array([])
    rng: np.random.Generator = np.random.default_rng()
    temperature: float = 1
    moves_per_sweep: int = 1
    observables: list = []

    def __init__(self, L_x: int, L_y: int):
        self.parameters = {
            'L_x': L_x,
            'L_y': L_y,
        }
        self.L_x = L_x
        self.L_y = L_y

        self.label = f'LoopModel2D {L_x}x{L_y}'
        self.spin_shape = (2*L_x, 2*L_y)
        self.bond_shape = (2*L_x, 2*L_y)

        # Default spin config.
        self.spins = np.zeros(self.spin_shape, dtype=int)
        self.spins[::2, 1::2] = 1
        self.spins[1::2, ::2] = 1

        # Default disorder config.
        self.disorder = np.zeros(self.bond_shape, dtype=int)
        self.disorder[::2, ::2] = 1

        # Default couplings config.
        self.couplings = np.zeros(self.bond_shape, dtype=float)
        self.couplings[::2, ::2] = 1

        self.rng = np.random.default_rng()
        self.temperature = 1.0
        self.moves_per_sweep = self.n_spins
        self.observables = [
            Energy(),
            Magnetization(),
            Susceptibility0(),
            Susceptibilitykmin(),
            WilsonLoop2D(self)
        ]

    @property
    def spin_index(self) -> Iterator[Tuple[int, int]]:
        positions = product(range(2), range(self.L_x), range(self.L_y))
        for normal, i, j in positions:
            if normal == 0:
                yield (2*i, 2*j + 1)
            else:
                yield (2*i + 1, 2*j)

    @property
    def bond_index(self) -> Iterator[Tuple[int, int]]:
        positions = product(range(self.L_x), range(self.L_y))
        for i, j in positions:
            yield (2*i, 2*j)

    def total_energy(self):
        energy = 0.0
        size_x, size_y = self.spin_shape

        for x, y in self.bond_index:
            disorder = self.disorder[x, y]
            coupling = self.couplings[x, y]
            spin_x1 = self.spins[x, (y + 1) % size_y]
            spin_x0 = self.spins[x, (y - 1) % size_y]
            spin_y1 = self.spins[(x + 1) % size_x, y]
            spin_y0 = self.spins[(x - 1) % size_x, y]
            energy -= coupling*disorder*spin_x1*spin_x0*spin_y1*spin_y0

        return energy

    def delta_energy(self, site) -> float:
        energy = 0.0
        x, y = site

        size_x, size_y = self.spin_shape

        # Spin on edge normal to x
        if (x % 2, y % 2) == (0, 1):
            bond_v_1 = self.disorder[x, (y + 1) % size_y]
            coupling_v_1 = self.couplings[x, (y + 1) % size_y]
            spin_e_1x1 = self.spins[x, (y + 2) % size_y]
            spin_e_1y1 = self.spins[(x + 1) % size_x, (y + 1) % size_y]
            spin_e_1y0 = self.spins[(x - 1) % size_x, (y + 1) % size_y]

            bond_v_0 = self.disorder[x, (y - 1) % size_y]
            coupling_v_0 = self.couplings[x, (y - 1) % size_y]
            spin_e_0x0 = self.spins[x, (y - 2) % size_y]
            spin_e_0y1 = self.spins[(x + 1) % size_x, (y - 1) % size_y]
            spin_e_0y0 = self.spins[(x - 1) % size_x, (y - 1) % size_y]

            energy = 2*self.spins[x, y]*(
                coupling_v_1*bond_v_1*spin_e_1x1*spin_e_1y0*spin_e_1y1
                + coupling_v_0*bond_v_0*spin_e_0x0*spin_e_0y0*spin_e_0y1
            )

        # Spin on edge normal to y
        else:
            bond_v_1 = self.disorder[(x + 1) % size_x, y]
            coupling_v_1 = self.couplings[(x + 1) % size_x, y]
            spin_e_1x1 = self.spins[(x + 1) % size_x, (y + 1) % size_y]
            spin_e_1x0 = self.spins[(x + 1) % size_x, (y - 1) % size_y]
            spin_e_1y1 = self.spins[(x + 2) % size_x, y]

            bond_v_0 = self.disorder[(x - 1) % size_x, y]
            coupling_v_0 = self.couplings[(x - 1) % size_x, y]
            spin_e_0x1 = self.spins[(x - 1) % size_x, (y + 1) % size_y]
            spin_e_0x0 = self.spins[(x - 1) % size_x, (y - 1) % size_y]
            spin_e_0y0 = self.spins[(x - 2) % size_x, y]

            energy = 2*self.spins[x, y]*(
                coupling_v_1*bond_v_1*spin_e_1x1*spin_e_1x0*spin_e_1y1
                + coupling_v_0*bond_v_0*spin_e_0x1*spin_e_0x0*spin_e_0y0
            )

        return energy

    @property
    def n_bonds(self) -> int:
        """Number of terms in the Hamiltonian whose sign can be flipped."""
        return self.L_x*self.L_y

    @property
    def n_spins(self) -> int:
        """Number of spins in the Hamiltonian."""
        return 2*self.L_x*self.L_y

    def init_spins(self, spins: Optional[np.ndarray] = None):
        """Initialize the spins. Random if None given."""
        if spins is not None:
            self.spins = spins
        else:
            L_x = self.L_x
            L_y = self.L_y
            self.spins = np.zeros(self.spin_shape, dtype=int)
            rand_spins_x = self.rng.integers(0, 2, size=(L_x, L_y))*2 - 1
            rand_spins_y = self.rng.integers(0, 2, size=(L_x, L_y))*2 - 1

            self.spins[::2, 1::2] = rand_spins_x
            self.spins[1::2, ::2] = rand_spins_y

    def init_disorder(self, disorder: Optional[np.ndarray] = None):
        if disorder is not None:
            self.disorder = disorder
        else:
            self.disorder = np.zeros(self.bond_shape, dtype=int)
            self.disorder[::2, ::2] = 1

    def random_move(self) -> Tuple[int, int]:
        """Random edge to propose flipping."""
        L_x = self.L_x
        L_y = self.L_y
        i = self.rng.integers(0, L_x)
        j = self.rng.integers(0, L_y)
        edge = self.rng.integers(0, 2)
        if edge == 0:
            x, y = 2*i, 2*j + 1
        else:
            x, y = 2*i + 1, 2*j
        return x, y

    def update(self, move):
        """Update spins with move."""
        self.spins[move] *= -1


class WilsonLoop2D(VectorObservable):
    label: str = 'Wilson Loop'

    def __init__(self, spin_model):
        L = min(spin_model.spins.shape)
        eps = 1e-6  # useful to round .5 to 1
        self.n_wilson_loops = int(np.round(L / 4 + eps))
        self.reset()

    @property
    def size(self):
        return self.n_wilson_loops

    def evaluate(self, spin_model) -> np.ndarray:
        value = np.ones(self.n_wilson_loops)
        spins = spin_model.spins

        for i in range(self.n_wilson_loops):
            bottom_row = np.prod(spins[2:2*i+3:2, 1])
            top_row = np.prod(spins[2:2*i+3:2, 2*i+3])
            left_col = np.prod(spins[1, 2:2*i+3:2])
            right_col = np.prod(spins[2*i+3, 2:2*i+3:2])
            value[i] = left_col * right_col * top_row * bottom_row

        return value


class LoopModel2DIidDisorder(DisorderModel):

    def generate(self, model_params, disorder_params):
        L_x = model_params['L_x']
        L_y = model_params['L_y']
        p = disorder_params['p']

        model = LoopModel2D(L_x, L_y)
        signs = np.ones(model.n_bonds, dtype=int)
        signs[self.rng.random(model.n_bonds) < p] = -1
        model.disorder[::2, ::2] = signs.reshape((L_x, L_y))

        disorder = model.disorder.copy()
        return disorder
