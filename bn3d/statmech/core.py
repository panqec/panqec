import os
from typing import List, Dict, Any
import json
from itertools import product
from pprint import pprint
import numpy as np
from .controllers import DataManager
from .config import DISORDER_MODELS


def generate_input_entries(ranges) -> List[Dict[str, Any]]:
    """Generate inputs and disorders from range spec"""
    entries = []
    for spec in ranges:
        iterates = product(
            spec['spin_model_params'], spec['disorder_params']
        )
        disorder_model_class = DISORDER_MODELS[spec['disorder_model']]
        for spin_model_params, disorder_params in iterates:
            for i_disorder in range(spec['n_disorder']):
                rng = np.random.default_rng(seed=i_disorder)
                disorder_model = disorder_model_class(rng=rng)
                disorder = disorder_model.generate(
                    spin_model_params, disorder_params
                )
                for temperature in spec['temperature']:
                    entry = {
                        'spin_model': spec['spin_model'],
                        'spin_model_params': spin_model_params,
                        'disorder_model': spec['disorder_model'],
                        'disorder_model_params': disorder_params,
                        'temperature': temperature,
                        'disorder': disorder.tolist()
                    }
                    entries.append(entry)
    return entries


def generate_inputs(data_dir):
    """Generate inputs using the targets.json file."""
    targets_json = os.path.join(data_dir, 'targets.json')
    with open(targets_json) as f:
        targets = json.load(f)

    print(f'Generating inputs and disorder configs from {targets_json}')
    pprint(targets)

    inputs = generate_input_entries(targets['ranges'])
    data_manager = DataManager(data_dir)
    data_manager.save('inputs', inputs)


def start_sampling(data_dir):
    print(f'Starting to sample up to in {data_dir}')
