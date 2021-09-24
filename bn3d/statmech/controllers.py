from typing import List
from .model import SpinModel


class SimpleController:

    models: List[SpinModel] = []
    results: List[dict] = []

    def __init__(self):
        self.models = []

    def run_models(self, tau: int):
        for model in self.models:
            model.sample(2**(tau - 1))
