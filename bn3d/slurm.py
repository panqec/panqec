"""
Utilities for automating slurm jobs.

:Author:
    Eric Huang
"""

import os
from typing import Dict
from glob import glob
from .config import SLURM_DIR, SBATCH_TEMPLATE


def generate_sbatch(
    n_trials: int = 1000,
    partition: str = 'defq',
    time: str = '01:00:00'
):
    """Generate sbatch files."""
    input_dir = os.path.join(SLURM_DIR, 'inputs')
    sbatch_dir = os.path.join(SLURM_DIR, 'sbatch')

    with open(SBATCH_TEMPLATE) as f:
        template_text = f.read()

    input_files = glob(os.path.join(input_dir, '*.json'))
    for input_file in input_files:
        name = os.path.splitext(os.path.split(input_file)[-1])[0]
        sbatch_file = os.path.join(sbatch_dir, f'{name}.sbatch')
        replacement: Dict[str, str] = {
            'partition': partition,
            'job_name': name,
            'nodes': '3',
            'output': f'slurm/out/{name}',
            'time': time,
            'input_dir': 'slurm/inputs',
            'input_name': name,
            'n_trials': str(n_trials)
        }
        for field, value in replacement.items():
            modified_text = template_text.replace('${%s}' % field, value)
        with open(sbatch_file, 'w') as f:
            f.write(modified_text)
