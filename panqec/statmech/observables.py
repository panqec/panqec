import numpy as np
from .model import ScalarObservable


class Susceptibility0(ScalarObservable):
    label: str = 'Susceptibility0'

    def __init__(self):
        self.reset()

    def evaluate(self, spin_model) -> float:
        value = float(np.abs(np.sum(spin_model.spins))**2/spin_model.n_spins)
        return value


class Magnetization(ScalarObservable):
    label: str = 'Magnetization'

    def __init__(self):
        self.reset()

    def evaluate(self, spin_model) -> float:
        value = float(np.sum(spin_model.spins)/spin_model.n_spins)
        return value


class Susceptibilitykmin(ScalarObservable):
    label: str = 'Susceptibilitykmin'

    def __init__(self):
        self.reset()

    def evaluate(self, spin_model) -> float:
        axis = int(np.argmax(spin_model.spin_shape))
        k_min = np.zeros(len(spin_model.spin_shape))
        k_min[axis] = 2*np.pi/spin_model.spin_shape[axis]
        positions = np.indices(spin_model.spin_shape)
        phases = np.exp(1j*np.tensordot(k_min, positions, axes=(0, 0)))
        value = float(
            np.abs(np.sum(spin_model.spins*phases))**2
            / spin_model.n_spins
        )
        return value


class Energy(ScalarObservable):
    label: str = 'Energy'

    def __init__(self):
        self.reset()

    def evaluate(self, spin_model) -> float:
        return float(spin_model.total_energy())
