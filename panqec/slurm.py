"""
Utilities for automating slurm jobs.

:Author:
    Eric Huang
"""

import os
import subprocess
import re
import datetime
from typing import Dict, Optional
from glob import glob
import numpy as np
from .config import (
    SLURM_DIR, SBATCH_TEMPLATE, SLURM_USERNAME, NIST_TEMPLATE
)
from .app import read_input_json, count_runs


def _delete_files_with_ext(folder_path, extension):
    """Delete files with extension in folder."""
    files = glob(os.path.join(folder_path, f'*.{extension}'))
    total_files = len(files)
    for file_path in files:
        print(f'Deleting {file_path}')
        os.remove(file_path)
    print(f'Deleted {total_files} .{extension} files')


def clear_sbatch_folder():
    """Delete all sbatch files in the sbatch folder."""
    sbatch_dir = os.path.join(SLURM_DIR, 'sbatch')
    _delete_files_with_ext(sbatch_dir, 'sbatch')
    _delete_files_with_ext(sbatch_dir, 'sh')


def clear_out_folder():
    """Delete all out files in the sbatch folder."""
    sbatch_dir = os.path.join(SLURM_DIR, 'out')
    _delete_files_with_ext(sbatch_dir, 'out')


def count_input_runs(name: str) -> Optional[int]:
    input_dir = os.path.join(SLURM_DIR, 'inputs')
    input_file = os.path.join(input_dir, f'{name}.json')
    n_runs = count_runs(input_file)
    return n_runs


def generate_sbatch_nist(
    name: str,
    n_trials: int,
    nodes: int,
    ntasks: int,
    cpus_per_task: int,
    mem: int,
    time: str,
    split: int,
    partition: str,
    cluster: str,
):
    """Generate sbatch files for NIST."""
    input_dir = os.path.join(SLURM_DIR, 'inputs')
    sbatch_dir = os.path.join(SLURM_DIR, 'sbatch')
    output_dir = os.path.join(SLURM_DIR, 'out')

    template_path = NIST_TEMPLATE
    if cluster == 'symmetry':
        template_path = SBATCH_TEMPLATE

    with open(template_path) as f:
        template_text = f.read()

    input_file = os.path.join(input_dir, f'{name}.json')
    if not os.path.exists(input_file):
        print(f'File {input_file} not found')
    else:
        print(f'Generating NIST sbatch files for {name}')
        run_count = count_runs(input_file)
        if split == 1 or run_count is None:
            split_label = ''
            options = ''
            _write_sbatch(
                sbatch_dir, template_text,
                name, output_dir, nodes, ntasks, cpus_per_task,
                mem, time, input_file, n_trials, options, split_label,
                partition
            )
        else:
            split = min(split, run_count)
            parts = np.array_split(range(run_count), split)
            sbatch_files = []
            for i_part, part in enumerate(parts):
                split_label = f'_{i_part}'
                start = part[0]
                n_runs = len(part)
                options = f' --start {start} --n_runs {n_runs}'
                sbatch_file = _write_sbatch(
                    sbatch_dir, template_text,
                    name, output_dir, nodes, ntasks, cpus_per_task,
                    mem, time, input_file, n_trials, options, split_label,
                    partition
                )
                sbatch_files.append(sbatch_file)

            write_submit_sh(name, sbatch_files)


def write_submit_sh(name, sbatch_files, flags=''):
    """Write script to submit all sbatch files for a given run."""
    sbatch_dir = os.path.join(SLURM_DIR, 'sbatch')
    sh_path = os.path.join(sbatch_dir, f'submit_{name}.sh')
    n_jobs = len(sbatch_files)
    lines = [
        '#!/bin/bash',
        f'echo "Submitting {n_jobs} jobs for {name}"'
    ]
    for sbatch_file in sbatch_files:
        lines.append(
            f'sbatch {flags} \'{sbatch_file}\''
        )
    lines.append('echo "Success! Happy waiting!"')
    sh_contents = '\n'.join(lines)
    with open(sh_path, 'w') as f:
        f.write(sh_contents)
    print(f'Run {sh_path} to submit all jobs')


def _write_sbatch(
    sbatch_dir: str, template_text: str,
    name: str, output_dir: str, nodes: int, ntasks: int, cpus_per_task: int,
    mem: int, time: str, input_file: str, n_trials: int, options: str,
    split_label: str, partition: str
) -> str:
    """Write sbatch file and return the path."""
    sbatch_file = os.path.join(
        sbatch_dir, f'{name}{split_label}.sbatch'
    )
    replacement: Dict[str, str] = {
        'partition': partition,
        'job_name': f'{name}{split_label}',
        'output': os.path.join(output_dir, f'{name}{split_label}.out'),
        'nodes': str(nodes),
        'ntasks': str(ntasks),
        'cpus_per_task': str(cpus_per_task),
        'mem': str(mem),
        'time': time,
        'input_file': input_file,
        'n_trials': str(n_trials),
        'options': options,
    }
    modified_text = template_text
    for field, value in replacement.items():
        field_key = '${%s}' % field
        assert field in modified_text, f'{field} missing from template'
        assert field_key in modified_text
        modified_text = modified_text.replace(field_key, value)
    with open(sbatch_file, 'w') as f:
        f.write(modified_text)
    print(f'Generated {sbatch_file}')
    return sbatch_file


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
        print(squeue_output.decode('utf-8'))
    except Exception:
        print('No squeue status available.')


def get_out_status():
    out_dir = os.path.join(SLURM_DIR, 'out')
    out_dir_glob = os.path.join(out_dir, '*.out')
    out_files = glob(out_dir_glob)
    if len(out_files) > 0:
        for out_file in out_files:
            tail_command = f'tail -n2 {out_file}'
            out_status = subprocess.Popen(
                tail_command.split(),
            )
            try:
                while True:
                    line = out_status.stdout.readline()
                    if not line:
                        break
            except AttributeError:
                print('')
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
        if wall_time + time_remaining > 0:
            progress = wall_time/(wall_time + time_remaining)
        else:
            progress = 0
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
