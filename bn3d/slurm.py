"""
Utilities for automating slurm jobs.

:Author:
    Eric Huang
"""

import os
from typing import Optional, List
from glob import glob
from .config import SLURM_DIR, SBATCH_TEMPLATE


def generate_sbatch(files: Optional[List[str]] = None):
    """Generate sbatch files."""
    input_dir = os.path.join(SLURM_DIR, 'inputs')
    sbatch_dir = os.path.join(SLURM_DIR, 'sbatch')

    with open(SBATCH_TEMPLATE) as f:
        template_text = f.read()

    input_files = glob(os.path.join(input_dir, '*.json'))
    for input_file in input_files:
        name = os.path.splitext(os.path.split(input_file)[-1])[0]
        sbatch_file = os.path.join(sbatch_dir, f'{name}.json')
        replacement = {
            'job_name': name,
        }
        for field, value in replacement.items():
            modified_text = template_text.replace('$' + field, value)
        with open(sbatch_file, 'w') as f:
            f.write(modified_text)
