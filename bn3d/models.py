"""
Models of codes and noise that extend qecsim classes.

:Author:
    Eric Huang
"""
import functools
from typing import Tuple
from qecsim.models.generic import SimpleErrorModel


class PauliErrorModel(SimpleErrorModel):

    direction: Tuple[float, float, float]

    def __init__(self, direction: Tuple[float, float, float]):
        self.direction = direction

    @property
    def label(self):
        return 'Pauli'

    @functools.lru_cache()
    def probability_distribution(self, probability: float) -> Tuple:
        r_x, r_y, r_z = self.direction
        p_i = 1 - probability
        p_x = r_x*probability
        p_y = r_y*probability
        p_z = r_z*probability
        return p_i, p_x, p_y, p_z
