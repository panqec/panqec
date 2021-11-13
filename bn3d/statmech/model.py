"""
Base classes for statistical mechanics simulations.
"""

from typing import Tuple, List, Optional, Any, Dict
from abc import ABCMeta, abstractmethod
import time
import numpy as np


class SpinModel(metaclass=ABCMeta):
    """Disordered Classical spin model."""

    temperature: float
    moves_per_sweep: int
    sweep_stats: Dict[str, Any]

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """Parameters used to create object."""

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
    def n_bonds(self) -> int:
        """Number of bonds."""

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
    def init_spins(self, spins: Optional[np.ndarray] = None):
        """Initialize the spins. Random if None given."""

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
    def total_energy(self) -> float:
        """Total energy of present state."""

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

    def seed_rng(self, seed):
        self.rng = np.random.default_rng(seed)

    @property
    @abstractmethod
    def observables(self) -> List:
        """Observables objects."""

    def observe(self):
        """Sample spins and record all observables."""
        for observable in self.observables:
            observable.record(self)

    def to_json(self) -> Dict[str, Any]:
        """Save to file."""
        data: Dict = {
            'label': self.label,
            'parameters': self.parameters,
            'spins': self.spins.tolist(),
            'disorder': self.disorder.tolist(),
            'rng': self.rng.__getstate__(),
            'temperature': self.temperature,
            'moves_per_sweep': self.moves_per_sweep,
            'observables': [
                observable.to_json()
                for observable in self.observables
            ],
        }
        return data

    def load_json(self, data: Dict[str, Any]):
        self.init_spins(np.array(data['spins']))
        self.init_disorder(np.array(data['disorder']))
        self.rng.__setstate__(data['rng'])
        self.temperature = self.temperature
        self.moves_per_sweep = data['moves_per_sweep']

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
        self.sweep_stats = stats
        return stats


class Observable(metaclass=ABCMeta):
    """Observable for a spin model."""

    total: Any
    total_2: Any
    total_4: Any
    count: int

    @abstractmethod
    def reset(self):
        """Reset record."""

    @property
    @abstractmethod
    def label(self) -> str:
        """Label for observable."""

    @abstractmethod
    def evaluate(self, spin_model: SpinModel):
        """Evaluate the observable for given spin model."""

    @abstractmethod
    def summary(self) -> Dict[str, Any]:
        """Return result dictionary that can be encoded into JSON"""

    def record(self, spin_model: SpinModel):
        """Evaluate the observable for given spin model and record it."""
        value = self.evaluate(spin_model)
        self.total += value
        self.total_2 += value**2
        self.total_4 += value**4
        self.count += 1

    def to_json(self) -> Dict[str, Any]:
        summary = self.summary()
        summary['label'] = self.label
        return summary


class ScalarObservable(Observable):

    total: float
    total_2: float
    total_4: float
    count: int

    def reset(self):
        self.total = 0.0
        self.total_2 = 0.0
        self.total_4 = 0.0
        self.count = 0

    def summary(self):
        return {
            'total': self.total,
            'total_2': self.total_2,
            'total_4': self.total_4,
            'count': self.count,
        }


class VectorObservable(Observable):

    total: np.ndarray
    total_2: np.ndarray
    total_4: np.ndarray
    count: int

    @property
    @abstractmethod
    def size(self) -> int:
        """Size of the vector (i.e. number of observables)"""

    def reset(self):
        self.total = np.zeros(self.size)
        self.total_2 = np.zeros(self.size)
        self.total_4 = np.zeros(self.size)
        self.count = 0

    def summary(self):
        return {
            'total': self.total.tolist(),
            'total_2': self.total_2.tolist(),
            'total_4': self.total_4.tolist(),
            'count': self.count,
        }


class DisorderModel(metaclass=ABCMeta):
    """Disorder generator representing a noise model."""

    rng: np.random.Generator

    def __init__(self, rng: np.random.Generator = None):
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()

    @abstractmethod
    def generate(self, model_params, disorder_params) -> np.ndarray:
        """Generate a disorder configuration."""
