from typing import Tuple, List, Optional, Dict
from .model import SpinModel, Observable
import numpy as np


class RandomBondIsingModel2D(SpinModel):
    """Random Bond Ising Model in 2D."""

    label: str = ''
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
        self.label = f'RandomBondIsingModel2D {L_x}x{L_y}'
        self.spin_shape = (L_x, L_y)
        self.bond_shape = (2, L_x, L_y)
        self.spins = np.ones(self.spin_shape, dtype=int)
        self.disorder = np.ones(self.bond_shape, dtype=int)
        self.couplings = np.ones(self.bond_shape, dtype=float)
        self.rng = np.random.default_rng()
        self.temperature = 1.0
        self.moves_per_sweep = self.n_spins
        self.observables = []

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

    def delta_energy(self, site) -> float:
        """Energy difference from flipping index."""
        energy_diff = 0.0
        L_x, L_y = self.spin_shape
        x, y = site
        x_p = (x + 1) % L_x
        x_m = (x - 1 + L_x) % L_x
        y_p = (x + 1) % L_y
        y_m = (y - 1 + L_y) % L_y

        # Two-body terms
        energy_diff -= 2*self.spins[x, y]*(
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
                self.disorder[0, x, y_m]*self.couplings[0, x, y_m]
                * self.spins[x, y_m]
            )
        )
        return energy_diff

    def update(self, move):
        """Update spins with move."""
        self.spins[move] *= -1


class Magnetization(Observable):
    label: str = ''
    total: float
    count: int

    def __init__(self):
        self.label = 'Magnetization'
        self.reset()

    def evaluate(self, spin_model) -> float:
        value = float(np.mean(spin_model.spins))
        return value

    def reset(self):
        self.total = 0.0
        self.count = 0

    def summary(self) -> Dict:
        return {
            'total': self.total,
            'count': self.count,
        }
