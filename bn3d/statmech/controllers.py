import os
from typing import List, Dict, Union
import json
from glob import glob
import numpy as np
from .model import SpinModel
from .config import SPIN_MODELS
from ..utils import hash_json


class DataManager:
    """Manager for data file system."""

    data_dir: str = ''
    subdirs: Dict[str, str] = {}

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self._make_directories()

    def _make_directories(self):
        """Make subdirectories if they don't exist."""
        os.makedirs(self.data_dir, exist_ok=True)
        subdir_names = ['inputs', 'results', 'chains']
        for name in subdir_names:
            self.subdirs[name] = os.path.join(self.data_dir, name)
            os.makedirs(self.subdirs[name], exist_ok=True)

    def load(self, subdir: str) -> List[dict]:
        """Find all saved json files and all load the data within."""
        file_paths = glob(
            os.path.join(self.subdirs[subdir], '*.json')
        )
        data_list: List[dict] = []
        for file_name in file_paths:
            with open(file_name) as f:
                entry = json.load(f)
                data_list.append(entry)
        return data_list

    def get_name(self, subdir, data: dict) -> str:
        """Unified enforcement of data file naming standard."""
        name = 'Untitled.json'
        if subdir == 'inputs':
            name = 'input_{}.json'.format(data['hash'])
        elif subdir == 'results':
            name = 'results_{}_seed{}_tau{}.json'.format(
                data['hash'], data['seed'], data['tau']
            )
        return name

    def get_path(self, subdir: str, data: dict) -> str:
        """Get path to file storing data."""
        return os.path.join(
            self.subdirs[subdir],
            self.get_name(subdir, data)
        )

    def save(self, subdir: str, data: Union[List[dict], dict]):
        """Save data as json in appropriate folder per naming standard."""

        if isinstance(data, list):
            entries = data
        else:
            entries = [data]

        for entry in entries:

            # Add hash to input disorder object if it doesn't have one.
            if subdir == 'inputs' and 'hash' not in entry:
                entry['hash'] = hash_json(entry)

            # Use the file path per naming convention.
            file_path = self.get_path(subdir, entry)
            with open(file_path, 'w') as f:
                json.dump(entry, f, sort_keys=True, indent=2)


class SimpleController:
    """Simple controller for running many chains."""

    hashes: List[str] = []
    models: List[SpinModel] = []
    results: List[dict] = []
    data_dir: str = ''
    subdirs: Dict[str, str] = {}
    data_manager: DataManager

    def __init__(self, data_dir: str):
        self.data_manager = DataManager(data_dir)
        self.hashes = []
        self.models = []

        inputs = self.data_manager.load('inputs')
        self.init_models(inputs)

    def run_models(self, tau: int):
        """Run models for 2^(tau - 1) sweeps."""
        for model in self.models:
            model.sample(2**(tau - 1))

    def init_models(self, inputs: list):
        """Instantiate SpinModel objects for each input."""
        self.data_manager.save('inputs', inputs)
        for entry in inputs:
            spin_model_class = SPIN_MODELS[entry['spin_model']]
            if isinstance(entry['spin_model_params'], dict):
                model = spin_model_class(**entry['spin_model_params'])
            else:
                model = spin_model_class(*entry['spin_model_params'])
            model.init_disorder(np.array(entry['disorder']))
            model.temperature = entry['temperature']
            self.hashes.append(entry['hash'])
            self.models.append(model)

    def run(self, max_tau, progress=None):
        """Run all models up to given tau."""
        if progress is None:
            def progress(x):
                return x

        iterates = list(zip(self.hashes, self.models))
        for key, model in progress(iterates):
            for tau in range(max_tau + 1):
                seed = 0
                results = {
                    'hash': key,
                    'seed': seed,
                    'tau': tau,
                    'observables': dict()
                }
                n_sweeps = 2**tau
                for observable in model.observables:
                    observable.reset()
                results['sweep_stats'] = model.sample(n_sweeps)
                results['spins'] = model.spins.tolist()
                for observable in model.observables:
                    results['observables'][observable.label] = (
                        observable.summary()
                    )
                self.data_manager.save('results', results)

    def get_summary(self) -> List[dict]:
        summary = self.data_manager.load('results')
        return summary
