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
