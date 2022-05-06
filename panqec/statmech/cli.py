import os
import re
import json
from shutil import copyfile
from glob import glob
from multiprocessing import Pool, cpu_count
import click
import pandas as pd
from .analysis import SimpleAnalysis
from .controllers import DataManager, DumbController
from .core import (
    start_sampling, generate_inputs, filter_input_hashes, monitor_usage,
)
from .config import SPIN_MODELS, DISORDER_MODELS
from panqec.utils import hash_json
BASE_DIR = os.path.abspath(os.path.dirname(__file__))


@click.group(invoke_without_command=True)
@click.pass_context
def statmech(ctx):
    """Routines for doing stat mech MCMC runs."""
    if not ctx.invoked_subcommand:
        print(ctx.get_help())


@click.command()
@click.argument('data_dir', type=click.Path(exists=True), required=True)
@click.argument('i_worker', type=click.INT, required=True)
@click.argument('n_workers', type=click.INT, required=True)
def assign_inputs(data_dir, i_worker, n_workers):
    """Print out path to input files for i_worker out of n_workers."""
    data_manager = DataManager(data_dir)

    temp_group_hashes = {
        entry['hash']: hash_json({
            k: v
            for k, v in entry.items()
            if k != 'temperature' and k != 'hash'
        })
        for entry in data_manager.load('inputs')
    }

    info_json = os.path.join(data_dir, 'info.json')
    with open(info_json) as f:
        info_dict = json.load(f)

    info = pd.DataFrame(info_dict)
    info.index.name = 'hash'
    info = info.reset_index()

    info['temp_group'] = info['hash'].map(temp_group_hashes)

    temp_groups = pd.concat([
        info.groupby('temp_group')['hash'].aggregate(list),
        info.groupby('temp_group')['mc_updates'].sum(),
        info.groupby('temp_group')['i_disorder'].min()
    ], axis=1).reset_index()
    temp_groups = temp_groups.sort_values(
        by=['mc_updates', 'i_disorder'], ascending=[False, True]
    )
    temp_groups = temp_groups.reset_index(drop=True)
    temp_groups['i_worker'] = temp_groups.index % n_workers + 1

    hashes = temp_groups[temp_groups['i_worker'] == i_worker]['hash'].sum()
    for value in hashes:
        print(value)


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
@click.argument('input_json', required=True)
def sample(input_json):
    """Perform single MCMC run."""

    # Make sure the file exists.
    assert os.path.isfile(input_json)
    data_dir = os.path.dirname(os.path.dirname(input_json))

    # Ensure the info.json file exists which tells it how many tau to run.
    assert os.path.exists(os.path.join(data_dir, 'info.json'))

    # Run just that single input.
    controller = DumbController(input_json)
    controller.run_all()


@click.command()
@click.argument('data_dir', required=True)
@click.option('--n_jobs', default=1, type=click.INT, show_default=True)
@click.option(
    '-m', '--monitor', default=False, is_flag=True, show_default=True
)
def sample_parallel(data_dir, n_jobs, monitor):
    """Perform MCMC runs in parallel."""

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
    if monitor:
        monitor_result = pool.starmap_async(
            monitor_usage, [(data_dir, i_job, n_jobs, 10)]
        )
    sampler_result = pool.starmap_async(start_sampling, arguments)
    pool.close()

    if monitor:
        monitor_result.get()
    sampler_result.get()


@click.command()
@click.argument('data_dir', required=True)
@click.option(
    '-t', '--targets', default=None, type=click.Path(exists=True),
    show_default=True,
    help='targets.json file, will look for targets.json if None exists'
)
def generate(data_dir, targets):
    """Generate inputs for MCMC."""
    os.makedirs(data_dir, exist_ok=True)
    targets_json = os.path.join(data_dir, 'targets.json')
    if targets is not None:
        copyfile(targets, targets_json)
    generate_inputs(data_dir)


@click.command()
@click.argument('sbatch_file', required=True)
@click.option('-d', '--data_dir', type=click.Path(exists=True), required=True)
@click.option('-n', '--n_array', default=6, type=click.INT, show_default=True)
@click.option('-q', '--queue', default='defq', type=str, show_default=True)
@click.option(
    '-w', '--wall_time', default='0-20:00', type=str, show_default=True
)
@click.option(
    '-s', '--split', default=1, type=click.INT, show_default=True
)
def pi_sbatch(sbatch_file, data_dir, n_array, queue, wall_time, split):
    """Generate PI-style sbatch file with parallel and array job."""
    template_file = os.path.join(
        os.path.dirname(os.path.dirname(BASE_DIR)), 'scripts',
        'statmech.sbatch'
    )
    with open(template_file) as f:
        text = f.read()

    inputs_dir = os.path.join(data_dir, 'inputs')
    assert os.path.isdir(inputs_dir), (
        f'{inputs_dir} missing, please create it and generate inputs'
    )
    name = os.path.basename(data_dir)
    replace_map = {
        '${DATADIR}': data_dir,
        '${TIME}': wall_time,
        '${NAME}': name,
        '${NARRAY}': str(n_array),
        '${QUEUE}': queue,
        '${SPLIT}': str(split),
    }
    for template_string, value in replace_map.items():
        text = text.replace(template_string, value)

    with open(sbatch_file, 'w') as f:
        f.write(text)
    print(f'Wrote to {sbatch_file}')


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


@click.command()
@click.argument('data_dir', required=True)
def get_progress(data_dir):
    info_json = os.path.join(data_dir, 'info.json')
    results_dir = os.path.join(data_dir, 'results')
    with open(info_json) as f:
        info_dict = json.load(f)
    results_list = []
    for name in os.listdir(results_dir):
        match = re.search(r'results_tau(\d+)_([a-f0-9]+)_seed(\d+).gz', name)
        if match:
            results_list.append({
                'hash': match.group(2),
                'tau': int(match.group(1)),
                'seed': match.group(3),
                'time': os.path.getmtime(os.path.join(results_dir, name)),
            })
    results = pd.DataFrame(results_list)

    print(
        results.groupby('tau')['hash'].count().reset_index()
        .rename(columns={'hash': 'results'}).to_string(index=False)
    )

    updates = pd.concat(
        [
            results.groupby('hash')['tau'].max(),
            pd.Series(info_dict['mc_updates'], name='target_updates'),
            pd.Series(info_dict['max_tau'], name='target_tau'),
            pd.Series(
                results.groupby('hash')['time'].min(), name='start_time'
            ),
            pd.Series(results.groupby('hash')['time'].max(), name='end_time'),
        ],
        axis=1
    )
    updates['progress'] = 2.0**(updates['tau'] - updates['target_tau'])

    # Work out the overall progress.
    total_updates = updates['target_updates'].sum()
    completed_updates = (updates['progress']*updates['target_updates']).sum()
    overall_progress = completed_updates/total_updates

    first_update = pd.Timestamp(
        updates['start_time'].min(), unit='s', tz='UTC'
    )
    last_update = pd.Timestamp(updates['end_time'].max(), unit='s', tz='UTC')
    time_elapsed = last_update - first_update

    time_remaining = time_elapsed*(1/overall_progress - 1)
    estimated_finish = last_update + time_remaining

    total_time_required = time_elapsed/overall_progress
    print('Total {} required'.format(total_time_required.round('s')))

    # Correction to account for time till last update.
    time_now = pd.Timestamp.now(tz='UTC')
    actual_time_elapsed = time_now - first_update
    actual_time_remaining = estimated_finish - time_now

    print('{:.2f}%, {} elapsed, {} remaining'.format(
        overall_progress*100,
        actual_time_elapsed.round('s'),
        actual_time_remaining.round('s')
    ))


statmech.add_command(status)
statmech.add_command(sample)
statmech.add_command(sample_parallel)
statmech.add_command(generate)
statmech.add_command(models)
statmech.add_command(analyse)
statmech.add_command(clear)
statmech.add_command(get_progress)
statmech.add_command(assign_inputs)
statmech.add_command(pi_sbatch)
