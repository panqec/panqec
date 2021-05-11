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
