import os
from typing import List, Dict, Any, Tuple
import json
from itertools import product
from pprint import pprint
import numpy as np
from .controllers import DataManager, SimpleController
from .config import DISORDER_MODELS
from ..utils import hash_json


def generate_input_entries(
    ranges: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Generate inputs and disorders from range spec"""
    entries = []
    max_tau_dict = dict()
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
                    max_tau_dict[hash_json(entry)] = spec['max_tau']
    return entries, max_tau_dict


def generate_inputs(data_dir: str):
    """Generate inputs using the targets.json file."""
    targets_json = os.path.join(data_dir, 'targets.json')
    with open(targets_json) as f:
        targets = json.load(f)

    print(f'Generating inputs and disorder configs from {targets_json}')
    pprint(targets)

    inputs, max_tau_dict = generate_input_entries(targets['ranges'])
    data_manager = DataManager(data_dir)
    data_manager.save('inputs', inputs)

    info_json = {
        'max_tau': max_tau_dict,
    }
    with open(os.path.join(data_dir, 'info.json'), 'w') as f:
        json.dump(info_json, f)


def start_sampling(data_dir):
    print(f'Starting to sample up in {data_dir}')
    controller = SimpleController(data_dir)
    controller.run()
