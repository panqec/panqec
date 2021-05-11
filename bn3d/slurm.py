"""
Utilities for automating slurm jobs.

:Author:
    Eric Huang
"""

import os
import subprocess
import re
import datetime
from typing import Dict
from glob import glob
from .config import SLURM_DIR, SBATCH_TEMPLATE, SLURM_USERNAME
from .app import read_input_json


def generate_sbatch(
    n_trials: int,
    partition: str,
    time: str,
    cores: int,
):
    """Generate sbatch files."""
    input_dir = os.path.join(SLURM_DIR, 'inputs')
    sbatch_dir = os.path.join(SLURM_DIR, 'sbatch')
    output_dir = os.path.join(SLURM_DIR, 'out')

    with open(SBATCH_TEMPLATE) as f:
        template_text = f.read()

    input_files = glob(os.path.join(input_dir, '*.json'))
    print(input_dir)
    print(f'Found {len(input_files)} input files')
    print('\n'.join(input_files))
    for input_file in input_files:
        name = os.path.splitext(os.path.split(input_file)[-1])[0]
        print(f'Generating sbatch for {name}')
        sbatch_file = os.path.join(sbatch_dir, f'{name}.sbatch')
        replacement: Dict[str, str] = {
            'partition': partition,
            'job_name': name,
            'nodes': str(cores),
            'output': os.path.join(output_dir, f'{name}.out'),
            'time': time,
            'input_file': input_file,
            'n_trials': str(n_trials)
        }
        modified_text = template_text
        for field, value in replacement.items():
            field_key = '${%s}' % field
            assert field in modified_text
            assert field_key in modified_text
            modified_text = modified_text.replace(field_key, value)
        with open(sbatch_file, 'w') as f:
            f.write(modified_text)
        print(f'Generated {sbatch_file}')


def get_status():
    """Get status of running jobs on slurm."""
    get_squeue_status()
    get_out_status()
    get_results_status()


def get_squeue_status():
    try:
        status_command = 'squeue'
        if SLURM_USERNAME is not None:
            status_command = f'squeue -u {SLURM_USERNAME}'
        squeue = subprocess.Popen(
            status_command.split(),
            stdout=subprocess.PIPE
        )
        squeue_output, squeue_error = squeue.communicate()
        print(squeue_output)
    except Exception:
        print('No squeue status available.')


def get_out_status():
    out_dir = os.path.join(SLURM_DIR, 'out')
    out_dir_glob = os.path.join(out_dir, '*.out')
    if len(glob(out_dir_glob)) > 0:
        tail_command = f'tail -n2 {out_dir_glob}'
        out_status = subprocess.Popen(
            tail_command.split(),
            stdout=subprocess.PIPE
        )
        out_status_output, out_status_error = out_status.communicate()
        print(out_status_output)
    else:
        print('No slurm .out files available')


def get_results_status():
    sbatch_files = glob(os.path.join(SLURM_DIR, 'sbatch', '*.sbatch'))
    input_files = []
    n_trials_list = []
    for sbatch_file in sbatch_files:
        with open(sbatch_file) as f:
            for line in f.readlines():
                match = re.search(r'--file (.*json)', line)
                if match:
                    input_files.append(os.path.abspath(match.group(1)))

                    n_trials = 100
                    match_trials = re.search(r'--trials (\d+)', line)
                    if match:
                        n_trials = int(match_trials.group(1))
                    n_trials_list.append(n_trials)
    for input_file, n_trials in zip(input_files, n_trials_list):
        batch_sim = read_input_json(input_file)
        batch_sim.load_results()
        print(batch_sim.label)
        wall_time = batch_sim.wall_time
        time_remaining = batch_sim.estimate_remaining_time(n_trials)
        progress = wall_time/(wall_time + time_remaining)
        print('    target n_trials =', n_trials)
        print(
            '    wall time =',
            str(datetime.timedelta(seconds=wall_time))
        )
        print(
            '    remaining time =',
            str(datetime.timedelta(seconds=time_remaining))
        )
        print(f'    progress = {progress*100:.2f}%')
