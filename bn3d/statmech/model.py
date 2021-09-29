"""
Base classes for statistical mechanics simulations.
"""

from typing import Tuple, List, Optional, Any, Dict
from abc import ABCMeta, abstractmethod
import time
import numpy as np


class SpinModel(metaclass=ABCMeta):
    """Disordered Classical spin model."""

    @property
    @abstractmethod
    def spin_shape(self) -> Tuple[int, ...]:
        """Shape of spin array."""

    @property
    @abstractmethod
    def bond_shape(self) -> Tuple[int, ...]:
        """Shape of spin array."""

    @property
    @abstractmethod
    def n_spins(self) -> int:
        """Size of model."""

    @property
    @abstractmethod
    def moves_per_sweep(self) -> int:
        """Number of MCMC moves to test per sweep."""

    @property
    @abstractmethod
    def n_bonds(self) -> int:
        """Number of bonds."""

    @property
    @abstractmethod
    def temperature(self) -> float:
        """Temperature of system in natural units."""

    @property
    @abstractmethod
    def disorder(self) -> np.ndarray:
        """Disorder of the model."""

    @property
    @abstractmethod
    def spins(self) -> np.ndarray:
        """State of the spins."""

    @property
    @abstractmethod
    def couplings(self) -> np.ndarray:
        """Coupling coefficients of the model."""

    @abstractmethod
    def init_disorder(self, disorder: Optional[np.ndarray] = None):
        """Initialize the disorder."""

    @property
    @abstractmethod
    def label(self) -> str:
        """Label to describe spin model."""

    @abstractmethod
    def delta_energy(self, move: Tuple) -> float:
        """Energy difference from apply moving."""

    @abstractmethod
    def random_move(self) -> Tuple:
        """Get a random move."""

    @abstractmethod
    def update(self, move: Tuple):
        """Update spin state with move."""

    @property
    @abstractmethod
    def rng(self) -> np.random.Generator:
        """Random number generator."""

    @property
    @abstractmethod
    def observables(self) -> List:
        """Observables objects."""

    def observe(self):
        """Sample spins and record all observables."""
        for observable in self.observables:
            observable.record(self)

    def sample(self, sweeps: int) -> dict:
        """Run MCMC sampling for given number of sweeps."""
        stats: Dict[str, Any] = {
            'acceptance': 0,
            'total': 0,
        }
        stats['start_time'] = time.time()
        for t in range(sweeps):
            for i_update in range(self.moves_per_sweep):
                move = self.random_move()
                if (
                    self.temperature*self.rng.exponential()
                    > self.delta_energy(move)
                ):
                    self.update(move)
                    stats['acceptance'] += 1
                stats['total'] += 1
            self.observe()
        stats['run_time'] = time.time() - stats['start_time']
        return stats


class Observable(metaclass=ABCMeta):
    """Observable for a spin model."""

    total: Any
    count: int

    @abstractmethod
    def reset(self):
        """Reset record."""

    @property
    @abstractmethod
    def label(self):
        """Label for observable."""

    @abstractmethod
    def evaluate(self, spin_model: SpinModel):
        """Evaluate the observable for given spin model."""

    def record(self, spin_model: SpinModel):
        """Evaluate the observable for given spin model and record it."""
        value = self.evaluate(spin_model)
        self.total += value
        self.count += 1

    def summary(self) -> Dict:
        return {
            'total': self.total,
            'count': self.count,
        }


class ScalarObservable(Observable):

    total: float
    count: int

    def reset(self):
        self.total = 0.0
        self.count = 0


class DisorderModel(metaclass=ABCMeta):
    """Disorder generator representing a noise model."""

    @abstractmethod
    def generate(self, model_params, disorder_params) -> np.ndarray:
        """Generate a disorder configuration."""
