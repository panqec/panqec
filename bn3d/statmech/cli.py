import os
import json
from glob import glob
from multiprocessing import Pool, cpu_count
import click
import pandas as pd
from .analysis import SimpleAnalysis
from .controllers import DataManager
from .core import (
    start_sampling, generate_inputs, filter_input_hashes, monitor_usage
)
from .config import SPIN_MODELS, DISORDER_MODELS


@click.group(invoke_without_command=True)
@click.pass_context
def statmech(ctx):
    """Routines for doing stat mech MCMC runs."""
    if not ctx.invoked_subcommand:
        print(ctx.get_help())


@click.command()
@click.argument('data_dir', required=True)
def analyse(data_dir):
    """Perform data analysis to extract useful stats."""

    print(f'Analysing {data_dir}')
    analysis = SimpleAnalysis(data_dir)
    analysis.analyse()

    analysis_dir = os.path.join(data_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)

    estimates_pkl = os.path.join(analysis_dir, 'estimates.pkl')
    results_pkl = os.path.join(analysis_dir, 'results.pkl')
    inputs_pkl = os.path.join(analysis_dir, 'inputs.pkl')
    analysis_json = os.path.join(analysis_dir, 'analysis.json')

    analysis.estimates.to_pickle(estimates_pkl)
    analysis.results_df.to_pickle(results_pkl)
    analysis.inputs_df.to_pickle(inputs_pkl)

    with open(analysis_json, 'w') as f:
        json.dump({
            'observable_names': analysis.observable_names,
            'independent_variables': analysis.independent_variables,
            'run_time_constants': analysis.run_time_constants,
        }, f, sort_keys=True, indent=2)


@click.command()
@click.argument('data_dir', required=True)
def status(data_dir):
    """Show the status of running jobs."""
    print(f'Showing statmech status in {data_dir}')

    dm = DataManager(data_dir)
    for subdir in dm.subdirs.keys():
        print('{}: {} files'.format(subdir, dm.count(subdir)))

    info_json = os.path.join(data_dir, 'info.json')
    if os.path.isfile(info_json):
        with open(info_json) as f:
            info_dict = json.load(f)
    analysis = SimpleAnalysis(data_dir)
    analysis.combine_inputs()
    analysis.combine_results()
    analysis.calculate_run_time_stats()

    # Estimated time assuming all inputs are run to the same max_tau and
    # the highest n_disorder, which may be an overestimate.
    estimated_time = pd.to_timedelta(
        analysis.estimate_run_time(
            dm.load('inputs'),
            max(info_dict['max_tau'].values()),
            max(info_dict['i_disorder'].values()) + 1,
        ),
        unit='s'
    )
    actual_time = pd.to_timedelta(
        analysis.run_time_stats['total_time'].sum(),
        unit='s'
    )
    print(f'Estimated CPU time {estimated_time}')
    print(f'Actual CPU time {actual_time}')
    progress = float(
        100*actual_time.total_seconds()/estimated_time.total_seconds()
    )
    print(f'Progress {progress:.2f}%')


@click.command()
@click.argument('data_dir', required=True)
@click.option('--n_jobs', default=1, type=click.INT, show_default=True)
def sample(data_dir, n_jobs):
    """Perform MCMC runs.."""

    # If running as array job slurm, determine what the task ID is.
    i_job = int(os.getenv('SLURM_ARRAY_TASK_ID', default=1))
    i_job = i_job - 1

    n_cpu = cpu_count()
    arguments = []
    for i_process in range(n_cpu):
        input_hashes = filter_input_hashes(
            data_dir, i_process, n_cpu, i_job, n_jobs
        )
        arguments.append((data_dir, input_hashes))

    print(f'Sampling over {n_cpu} CPUs for array job {i_job} out of {n_jobs}')
    pool = Pool(processes=n_cpu + 1)
    monitor_result = pool.starmap_async(
        monitor_usage, [(data_dir, i_job, n_jobs, 10)]
    )
    sampler_result = pool.starmap_async(start_sampling, arguments)
    pool.close()

    monitor_result.get()
    sampler_result.get()


@click.command()
@click.argument('data_dir', required=True)
def generate(data_dir):
    """Generate inputs for MCMC."""
    generate_inputs(data_dir)


@click.command()
@click.argument('data_dir', required=True)
def clear(data_dir):
    """Clear inputs and results."""
    subdirs = ['inputs', 'results', 'models', 'runs', 'logs']
    file_path_list = []
    for subdir in subdirs:
        file_path_list += glob(os.path.join(data_dir, 'inputs', '*'))

    loose_files = ['info.json', 'estimates.pkl']
    for file_name in loose_files:
        file_path = os.path.join(data_dir, file_name)
        if os.path.isfile(file_path):
            file_path_list.append(file_path)

    if click.confirm(f'Delete {len(file_path_list)} files?', default=False):
        for file_path in file_path_list:
            os.remove(file_path)


@click.command()
def models():
    """List available models."""
    print('Spin models')
    for spin_model in SPIN_MODELS:
        print(f'  {spin_model}')
    print('Disorder models')
    for disorder_model in DISORDER_MODELS:
        print(f'  {disorder_model}')


statmech.add_command(status)
statmech.add_command(sample)
statmech.add_command(generate)
statmech.add_command(models)
statmech.add_command(analyse)
statmech.add_command(clear)
