"""
Utilities for automating slurm jobs.

:Author:
    Eric Huang
"""

import os
import subprocess
import re
import datetime
from typing import Optional
from glob import glob
from .config import (
    SLURM_DIR, SLURM_USERNAME
)
from .simulation import read_input_json, count_runs


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
