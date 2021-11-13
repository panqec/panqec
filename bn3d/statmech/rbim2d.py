from typing import Tuple, Optional, Dict, Any
from itertools import product
from .model import SpinModel, DisorderModel, VectorObservable
from .observables import (
    Energy, Magnetization, Susceptibility0, Susceptibilitykmin
)
import numpy as np


class RandomBondIsingModel2D(SpinModel):
    """Random Bond Ising Model in 2D."""

    label: str = ''
    parameters: Dict[str, Any] = {}
    spin_shape: Tuple[int, int] = (0, 0)
    bond_shape: Tuple[int, int, int] = (0, 0, 0)
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
        self.label = f'RandomBondIsingModel2D {L_x}x{L_y}'
        self.spin_shape = (L_x, L_y)
        self.bond_shape = (2, L_x, L_y)
        self.spins = np.ones(self.spin_shape, dtype=int)
        self.disorder = np.ones(self.bond_shape, dtype=int)
        self.couplings = np.ones(self.bond_shape, dtype=float)
        self.rng = np.random.default_rng()
        self.temperature = 1.0
        self.moves_per_sweep = self.n_spins
        self.observables = [
            Energy(),
            Magnetization(),
            Susceptibility0(),
            Susceptibilitykmin(),
            WilsonLoop2D(self),
        ]

    @property
    def n_spins(self) -> int:
        """Size of model."""
        L_x, L_y = self.spin_shape
        return L_x*L_y

    @property
    def n_bonds(self) -> int:
        """Number of bonds."""
        return int(np.prod(self.bond_shape))

    def random_move(self) -> Tuple[int, int]:
        L_x, L_y = self.spin_shape
        x = self.rng.integers(0, L_x)
        y = self.rng.integers(0, L_y)
        return x, y

    def init_spins(self, spins: Optional[np.ndarray] = None):
        """Initialize the spins. Random if None given."""
        if spins is not None:
            self.spins = spins
        else:
            self.spins = self.rng.integers(0, 2, size=self.spin_shape)*2 - 1

    def init_disorder(self, disorder: Optional[np.ndarray] = None):
        """Initialize the disorder. Random if None given."""
        if disorder is not None:
            self.disorder = disorder

    def total_energy(self) -> float:
        """Total energy of spin state."""
        energy = 0.0
        L_x, L_y = self.spin_shape
        for axis, x, y in product(range(2), range(L_x), range(L_y)):
            disorder = self.disorder[axis, x, y]
            coupling = self.couplings[axis, x, y]
            spin_0 = self.spins[x, y]
            if axis == 0:
                spin_1 = self.spins[(x + 1) % L_x, y]
            else:
                spin_1 = self.spins[x, (y + 1) % L_y]
            energy -= coupling*disorder*spin_0*spin_1
        return energy

    def delta_energy(self, site) -> float:
        """Energy difference from flipping index."""
        energy_diff = 0.0
        L_x, L_y = self.spin_shape
        x, y = site
        x_p = (x + 1) % L_x
        x_m = (x - 1) % L_x
        y_p = (y + 1) % L_y
        y_m = (y - 1) % L_y

        # Two-body terms
        energy_diff = 2*self.spins[x, y]*(
            (
                self.disorder[0, x, y]*self.couplings[0, x, y]
                * self.spins[x_p, y]
            )
            + (
                self.disorder[1, x, y]*self.couplings[1, x, y]
                * self.spins[x, y_p]
            )
            + (
                self.disorder[0, x_m, y]*self.couplings[0, x_m, y]
                * self.spins[x_m, y]
            )
            + (
                self.disorder[1, x, y_m]*self.couplings[1, x, y_m]
                * self.spins[x, y_m]
            )
        )
        return energy_diff

    def update(self, move):
        """Update spins with move."""
        self.spins[move] *= -1


class WilsonLoop2D(VectorObservable):
    label: str = 'Wilson Loop'

    def __init__(self, spin_model):
        L = min(spin_model.spin_shape)
        self.n_wilson_loops = int(np.ceil(L / 2))

        self.reset()

    @property
    def size(self):
        return self.n_wilson_loops

    def evaluate(self, spin_model) -> np.ndarray:
        value = np.ones(self.n_wilson_loops)
        spins = spin_model.spins

        for i in range(self.n_wilson_loops):
            bottom_row = np.prod(spins[0, :i+1])
            top_row = np.prod(spins[i, :i+1]) if i > 0 else 1
            left_col = np.prod(spins[1:i, 0]) if i >= 2 else 1
            right_col = np.prod(spins[1:i, i]) if i >= 2 else 1
            value[i] = left_col * right_col * top_row * bottom_row

        return value


class Rbim2DIidDisorder(DisorderModel):

    def generate(self, model_params, disorder_params):
        L_x = model_params['L_x']
        L_y = model_params['L_y']
        p = disorder_params['p']
        disorder = np.ones((2, L_x, L_y), dtype=int)
        disorder[self.rng.random((2, L_x, L_y)) < p] = -1
        return disorder
