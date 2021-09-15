"""
Base classes for statistical mechanics simulations.
"""

from typing import Tuple, List
from abc import ABCMeta, abstractmethod


class SpinModel(ABCMeta):
    """Disordered Classical spin model."""

    @property
    @abstractmethod
    def disorder(self):
        """Disorder of the model."""

    @property
    @abstractmethod
    def spins(self):
        """State of the spins."""

    @abstractmethod
    def init_disorder(self):
        """Initialize the disorder."""

    @property
    @abstractmethod
    def label(self):
        """Label to describe spin model.
        """

    @abstractmethod
    def delta_energy(self, index: Tuple[int]):
        """Energy difference from flipping index.
        """


class Observable(ABCMeta):
    """Observable for a spin model."""

    @property
    @abstractmethod
    def label(self):
        """Label for observable."""

    @abstractmethod
    def evaluate(self, spin_model: SpinModel):
        """Evaluate the observable for given spin model."""


class Sampler(ABCMeta):

    @property
    @abstractmethod
    def spin_model(self) -> SpinModel:
        """The spin model being sampled."""

    @property
    @abstractmethod
    def observables(self) -> List[Observable]:
        """Observables to sample."""
