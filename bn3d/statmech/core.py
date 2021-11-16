import os
from typing import List, Dict, Any, Tuple
import json
import time
import datetime
from itertools import product
from pprint import pprint
import numpy as np
import psutil
from .analysis import count_updates
from .controllers import DataManager, SimpleController
from .config import DISORDER_MODELS
from ..utils import hash_json


def generate_input_entries(
    ranges: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict]:
    """Generate inputs and disorders from range spec"""
    entries = []
    info_dict: Dict = {
        'max_tau': {},
        'i_disorder': {},
        'mc_updates': {},
    }
    for spec in ranges:
        iterates = product(
            spec['spin_model_params'], spec['disorder_params']
        )
        disorder_model_class = DISORDER_MODELS[spec['disorder_model']]
        for spin_model_params, disorder_params in iterates:
            for i_disorder in range(spec['n_disorder']):
                rng = np.random.default_rng(seed=i_disorder)
                disorder_model = disorder_model_class(rng=rng)  # type:ignore
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

                    # Information to be stored in info.json.
                    hash_key = hash_json(entry)
                    info_dict['max_tau'][hash_key] = spec['max_tau']
                    info_dict['i_disorder'][hash_key] = i_disorder

                    # Calculate the number of MCMC updates.
                    update_params = spin_model_params.copy()
                    update_params['tau'] = spec['max_tau'] + 1
                    info_dict['mc_updates'][hash_key] = count_updates(
                        spec['spin_model'], update_params
                    )
    return entries, info_dict


def generate_inputs(data_dir: str):
    """Generate inputs using the targets.json file in given directory."""
    targets_json = os.path.join(data_dir, 'targets.json')
    with open(targets_json) as f:
        targets = json.load(f)

    print(f'Generating inputs and disorder configs from {targets_json}')
    pprint(targets)

    inputs, info_dict = generate_input_entries(targets['ranges'])
    data_manager = DataManager(data_dir)
    data_manager.save('inputs', inputs)

    with open(os.path.join(data_dir, 'info.json'), 'w') as f:
        json.dump(info_dict, f)
    print(f'Saved {len(inputs)} disorder')


def start_sampling(data_dir, input_hashes=None) -> int:
    """Start MCMC sampling in given dir on list of input hashes."""
    print(f'Starting to sample in {data_dir}')
    controller = SimpleController(data_dir)
    if input_hashes:
        print(f'Filtering over {len(input_hashes)} inputs')
    controller.use_filter(input_hashes)
    controller.run()
    print('Runs complete')
    return 0


def filter_input_hashes(
    data_dir: str, i_process: int, n_processes: int, i_job: int, n_jobs: int
) -> List[str]:
    """Get list of filtered input hashes assigned to worker."""
    info_json = os.path.join(data_dir, 'info.json')
    with open(info_json) as f:
        info_dict = json.load(f)

    # Filter the hashes by converting the hex string to int and doing modulo
    # arithmetic so each worker gets roughly the same number of tasks.
    filtered_hashes: List[str] = [
        hash_key for hash_key in info_dict['i_disorder'].keys()
        if i_process*n_jobs + i_job
        == int(hash_key, 16) % (n_jobs*n_processes)
    ]
    return filtered_hashes


def monitor_usage(data_dir, i_job: int, n_jobs: int, interval: float = 60):
    """Continously monitor CPU usage."""
    os.makedirs(os.path.join(data_dir, 'logs'), exist_ok=True)
    ppid = os.getppid()
    log_file = os.path.join(data_dir, 'logs', f'{ppid}.txt')
    if not os.path.isfile(log_file):
        with open(log_file, 'w') as f:
            f.write(f'Log file for {ppid}\n')
    while True:
        cpu_usage = psutil.cpu_percent(percpu=True)
        mean_cpu_usage = np.mean(cpu_usage)
        n_cores = len(cpu_usage)
        time_now = datetime.datetime.now()
        message = (
            f'{time_now} CPU usage {mean_cpu_usage:.2f}% '
            f'({n_cores} cores)'
        )
        print(message, flush=True)
        with open(log_file, 'a') as f:
            f.write(message + '\n')
        time.sleep(interval)
