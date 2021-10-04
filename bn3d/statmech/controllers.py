"""
Classes for controlling MCMC chains in parallel and
managing the data produced.

:Author:
    Eric Huang
"""
import os
import re
from typing import List, Dict, Union, Any, Optional
import time
import uuid
import json
from glob import glob
import numpy as np
from .model import SpinModel
from .config import SPIN_MODELS
from ..utils import hash_json


class DataManager:
    """Manager for data file system.

    Kind of like a database, allows one to query json files (rows)
    with attributes (columns)
    stored in folders (tables)
    In the future we should use a real database like sqlite via an ORM like
    SQLAlchemy but for now this file-based storage is what we've got to still
    work with the clusters.
    """

    UNTITLED: str = 'Untitled.json'
    data_dir: str = ''
    subdirs: Dict[str, str] = {}

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self._make_directories()

    def _make_directories(self):
        """Make subdirectories if they don't exist."""
        self.subdirs = dict()
        os.makedirs(self.data_dir, exist_ok=True)
        subdir_names = ['inputs', 'results', 'models', 'queue', 'runs']
        for name in subdir_names:
            self.subdirs[name] = os.path.join(self.data_dir, name)
            os.makedirs(self.subdirs[name], exist_ok=True)

    def load(self, subdir: str, filters: Dict[str, Any] = {}) -> List[dict]:
        """Load saved data into list of dicts with optional filter."""
        file_paths = self.filter_files(subdir, filters)
        data_list: List[dict] = []
        for file_name in file_paths:
            with open(file_name) as f:
                entry = json.load(f)
                data_list.append(entry)
        return data_list

    def count(self, subdir: str) -> int:
        """Count the number of files stored."""
        dir_path = self.subdirs[subdir]
        return len(os.listdir(dir_path))

    def get_name(self, subdir: str, data: dict) -> str:
        """Unified enforcement of data file naming standard."""
        name = self.UNTITLED
        if subdir == 'inputs':
            name = 'input_{}.json'.format(data['hash'])
        elif subdir == 'results':
            name = 'results_tau{}_{}_seed{}.json'.format(
                data['tau'], data['hash'], data['seed']
            )
        elif subdir == 'models':
            name = 'model_{}.json'.format(data['hash'])
        elif subdir == 'runs':
            name = 'results_tau{}_{}_seed{}.json'.format(
                data['tau'], data['hash'], data['seed']
            )
        elif subdir == 'queue':
            name = 'queue_tau{}_{}_seed{}.json'.format(
                data['hash'], data['seed'], data['tau']
            )
        return name

    def get_params(self, subdir: str, name: str) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if subdir == 'inputs':
            match = re.search(r'input_([0-9a-f]+).json', name)
            if match:
                params = {
                    'hash': str(match.group(1)),
                }
        elif subdir == 'results':
            match = re.search(
                r'results_tau(\d+)_([0-9a-f]+)_seed(\d+).json', name
            )
            if match:
                params = {
                    'tau': int(match.group(1)),
                    'hash': str(match.group(2)),
                    'seed': int(match.group(3)),
                }
        elif subdir == 'models':
            match = re.search(r'model_([0-9a-f]+).json', name)
            if match:
                params = {
                    'hash': str(match.group(1)),
                }
        elif subdir == 'runs':
            match = re.search(r'run_([0-9a-f]+)_seed(\d+)_tau(\d+).json', name)
            if match:
                params = {
                    'tau': int(match.group(1)),
                    'hash': str(match.group(2)),
                    'seed': int(match.group(3)),
                }
        elif subdir == 'queue':
            match = re.search(
                r'task_tau(\d+)_([0-9a-f]+)_seed(\d+).json', name
            )
            if match:
                params = {
                    'tau': int(match.group(1)),
                    'hash': str(match.group(2)),
                    'seed': int(match.group(3)),
                }
        return params

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

    def filter_files(self, subdir: str, filters: Dict[str, Any]) -> List[str]:
        """Get list of file paths matching filter criterion."""

        # Try to see if expected path exists.
        expected_path: Optional[str] = None
        try:
            expected_path = self.get_path(subdir, filters)
            if expected_path == self.UNTITLED:
                expected_path = None
        except KeyError:
            expected_path = None

        # Return the list with path if expected path exists.
        if expected_path is not None and os.path.isfile(expected_path):
            filtered_paths = [expected_path]

        # Otherwise go looking through the entire subdirectory.
        else:
            file_paths = glob(os.path.join(self.subdirs[subdir], '*.json'))
            filtered_paths = []
            for file_path in file_paths:
                matches_filter = True
                file_name = os.path.split(file_path)[-1]
                file_params = self.get_params(subdir, file_name)
                if any(
                    key not in file_params.keys()
                    or file_params[key] != filters[key]
                    for key in filters.keys()
                ):
                    matches_filter = False

                if matches_filter:
                    filtered_paths.append(file_path)
        return filtered_paths

    def remove(self, subdir: str, filters: Dict[str, Any]):
        """Delete entries satisfying filter criterion."""
        files = self.filter_files(subdir, filters)
        if subdir == 'runs':
            for file_name in files:
                os.remove(file_name)


class SimpleController:
    """Simple controller for running many chains."""

    uuid: str = ''
    hashes: List[str] = []
    models: List[SpinModel] = []
    results: List[dict] = []
    data_dir: str = ''
    subdirs: Dict[str, str] = {}
    data_manager: DataManager

    def __init__(self, data_dir: str):
        self.uuid = uuid.uuid4().hex
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

    def single_run(
        self, model: SpinModel, input_hash: str, seed: int, tau: int
    ):
        """Single run of MCMC on given spin model.
        """
        results = {
            'hash': input_hash,
            'seed': seed,
            'tau': tau
        }
        observables_results: Dict[str, dict] = dict()
        n_sweeps = 2**tau
        for observable in model.observables:
            observable.reset()

        # Data to save in runs folder to let everyone else know
        # that this controller is about to start running a sample.
        run_data = {
            'hash': input_hash,
            'seed': seed,
            'tau': tau,
            'controller': self.uuid,
            'pid': os.getpid(),
            'start': time.time(),
        }
        self.data_manager.save('runs', run_data)

        # Do the MCMC sampling and obtain sweep stats.
        results['sweep_stats'] = model.sample(n_sweeps)

        results['spins'] = model.spins.tolist()
        for observable in model.observables:
            observables_results[observable.label] = (
                observable.summary()
            )
        results['observables'] = observables_results

        # Get a hash of the model and save it.
        model_json = model.to_json()
        model_hash = hash_json(model_json)
        model_json['hash'] = model_hash
        results['model'] = model_hash
        self.data_manager.save('models', model_json)

        # Save the results.
        self.data_manager.save('results', results)

        # Remove the run record in the runs folder so others know
        # the run has been completed.
        self.data_manager.remove('runs', {
            'hash': input_hash,
            'seed': seed,
            'tau': tau
        })

    def run(self, max_tau, progress=None):
        """Run all models up to given tau."""
        if progress is None:
            def progress(x):
                return x
        seed = 0

        for tau in range(max_tau + 1):
            iterates = list(zip(self.hashes, self.models))
            for input_hash, model in progress(iterates):

                # Find existing models and results before running.
                existing_results = self.data_manager.load('results', {
                    'hash': input_hash,
                    'seed': seed,
                    'tau': tau,
                })

                # Only proceed if there are no existing results.
                if len(existing_results) == 0:

                    # Whether the run for the previous tau has been done.
                    last_run_done = False

                    # Load the model state from the last tau if available.
                    if tau > 0:
                        previous_results = self.data_manager.load('results', {
                            'hash': input_hash,
                            'seed': seed,
                            'tau': tau - 1,
                        })
                        if previous_results:
                            last_run = previous_results[0]
                            model_hash = last_run['model']
                            model_json = self.data_manager.load('models', {
                                'hash': model_hash,
                            })[0]
                            model.load_json(model_json)
                            last_run_done = True

                    # Only proceed if run for previous tau complete or is the
                    # first run.
                    if last_run_done or tau == 0:

                        # Perform a single run.
                        self.single_run(model, input_hash, seed, tau)

                    else:

                        print('Missing file {}'.format(
                            self.data_manager.get_name('results', {
                                'hash': input_hash,
                                'seed': seed,
                                'tau': tau - 1
                            })[0]
                        ))

    def get_summary(self) -> List[dict]:
        """Get list of all results."""
        summary = self.data_manager.load('results')
        return summary
