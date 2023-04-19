import os
from typing import Optional, List
import click
import panqec
from tqdm import tqdm
import numpy as np
import json
from json.decoder import JSONDecodeError
import multiprocessing
import datetime
import time
import psutil
from .simulation import (
    run_file
)
from .config import CODES, ERROR_MODELS, DECODERS, PANQEC_DIR
from .slurm import (
    get_status, count_input_runs,
    clear_out_folder, clear_sbatch_folder
)
from .utils import (
    get_direction_from_bias_ratio, load_json, save_json, progress_bar
)
from panqec.gui import GUI
from glob import glob
from .usage import summarize_usage
from .analysis import Analysis


@click.group(invoke_without_command=True)
@click.version_option(version=panqec.__version__, prog_name='panqec')
@click.pass_context
def cli(ctx):
    """
    panqec - biased noise in 3D simulations.

    See panqec COMMAND --help for command-specific help.
    """
    if not ctx.invoked_subcommand:
        print(ctx.get_help())


@click.command()
@click.option('-p', '--port', 'port')
def start_gui(port: Optional[int]):
    gui = GUI()
    gui.run(port=port)


@click.command()
@click.pass_context
@click.option('-i', '--input_file', type=str)
@click.option('-o', '--output_file', type=click.STRING)
@click.option('-t', '--trials', default=100, type=click.INT, show_default=True)
def run(
    ctx,
    input_file: Optional[str],
    output_file: str,
    trials: int
):
    """Run a single job or run many jobs from input file."""
    if input_file is not None:
        run_file(
            os.path.abspath(input_file),
            os.path.abspath(output_file),
            trials,
            progress=tqdm
        )
    else:
        print(ctx.get_help())


@click.command()
@click.option('-d', '--data_dir')
@click.option(
    '-t', '--trials', default=1000, type=click.INT, show_default=True
)
@click.option(
    '-n', '--n_nodes', default=1, type=click.INT, show_default=True
)
@click.option(
    '-j', '--job_idx', default=1, type=click.INT, show_default=True
)
@click.option(
    '-c', '--n_cores', default=None, type=click.INT, show_default=True
)
@click.option(
    '--delete-existing', is_flag=True, default=False, show_default=True,
    help="Delete existing results folder in the data directory"
)
def run_parallel(
    data_dir: str,
    trials: int,
    n_nodes: int,
    job_idx: int,
    n_cores: Optional[int],
    delete_existing: bool,
    compressed_output: bool = True
):
    """Run panqec jobs in parallel"""

    input_dir = os.path.join(data_dir, "inputs")
    result_dir = os.path.join(data_dir, "results")
    logs_dir = os.path.join(data_dir, "logs")
    progress_dir = os.path.join(logs_dir, "progress")

    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(progress_dir, exist_ok=True)

    i_node = job_idx - 1

    assert 1 <= job_idx <= n_nodes, \
        f"job_id={job_idx} is invalid. It must be between 1 and {n_nodes}"

    n_cpu = multiprocessing.cpu_count()

    if not n_cores:
        n_cores = n_cpu

    assert n_cores <= n_cpu, \
        f"The number of cores requested ({n_cores}) is higher than" \
        f"the total number of cores ({n_cpu})"

    print(f"Running job {job_idx}/{n_nodes} on {n_cores} cores")

    n_tasks = n_nodes * n_cores

    print(f"Total number of tasks: {n_tasks}\n")

    list_inputs = glob(f"{input_dir}/*.json")

    print("List inputs", list_inputs)

    n_inputs = len(list_inputs)

    if n_inputs == 0:
        raise ValueError(f"No input files in {input_dir}")

    procs = []
    for i_core in range(n_cores):
        i_task = n_cores * i_node + i_core

        n_tasks_per_input = n_tasks // n_inputs

        i_input = i_task // n_tasks_per_input
        if i_input >= n_inputs:
            i_input = n_inputs - 1

        if i_input == n_inputs - 1:
            n_tasks_per_input = n_tasks_per_input + n_tasks % n_inputs

        i_task_in_input = i_task % n_tasks_per_input

        if i_input == n_inputs - 1:
            i_task_in_input = i_task - n_tasks // n_inputs * (n_inputs - 1)

        n_runs = trials // n_tasks_per_input

        if i_task_in_input == n_tasks_per_input - 1:
            n_runs += trials % n_runs

        filename = list_inputs[i_input]
        input_name = os.path.basename(filename)

        # Split the results over files results_1.json, results_2.json, etc.
        max_n_digits = len(str(n_tasks))
        result_json_file = os.path.abspath(os.path.join(
            result_dir,
            f"results_{str(i_task+1).zfill(max_n_digits)}.json"
        ))
        result_gz_file = result_json_file + ".gz"

        for f in [result_json_file, result_gz_file]:
            if delete_existing and os.path.exists(f):
                os.remove(f)

        if compressed_output:
            result_file = result_gz_file
        else:
            result_file = result_json_file

        log_file = os.path.abspath(os.path.join(
            progress_dir,
            f"progress_{str(i_task+1).zfill(max_n_digits)}.txt"
        ))
        print(f"{input_name}\t{n_runs}")

        input_file = os.path.abspath(os.path.join(input_dir, input_name))

        proc = multiprocessing.Process(
            target=run_file,
            args=(input_file, result_file, n_runs),
            kwargs={
                'progress': tqdm,
                'log_file': log_file
            }
        )
        procs.append(proc)
        proc.start()

    # complete the processes
    for proc in procs:
        proc.join()


@click.command()
@click.argument('model_type', required=False, type=click.Choice(
    ['codes', 'error_models', 'decoders'],
    case_sensitive=False
))
def ls(model_type=None):
    """List available codes, error models and decoders."""
    if model_type is None or model_type == 'codes':
        print('Codes:')
        print('\n'.join([
            '    ' + name for name in sorted(CODES.keys())
        ]))
    if model_type is None or model_type == 'error_models':
        print('Error Models (Noise):')
        print('\n'.join([
            '    ' + name for name in sorted(ERROR_MODELS.keys())
        ]))
    if model_type is None or model_type == 'decoders':
        print('Decoders:')
        print('\n'.join([
            '    ' + name for name in sorted(DECODERS.keys())
        ]))


def read_bias_ratios(eta_string: str) -> list:
    """Read bias ratios from comma separated string."""
    bias_ratios = []
    for s in eta_string.split(','):
        s = s.strip()
        if s == 'inf':
            bias_ratios.append(np.inf)
        elif float(s) % 1 == 0:
            bias_ratios.append(int(s))
        else:
            bias_ratios.append(float(s))
    return bias_ratios


def read_range_input(specification: str) -> List[float]:
    """Read range input string and return list."""
    values: List[float] = []
    if ':' in specification:
        parts = specification.split(':')
        min_value = float(parts[0])
        max_value = float(parts[1])
        step = 0.005
        if len(parts) == 3:
            step = float(parts[2])
        values = np.arange(min_value, max_value + step, step).tolist()
    elif ',' in specification:
        values = [float(s) for s in specification.split(',')]
    else:
        values = [float(specification)]
    return values


@click.command()
@click.option(
    '-o', '--overrides', type=click.Path(exists=True),
    default=None,
    help='Overrides specification .json file.'
)
@click.option(
    '-p', '--plot_dir', type=click.Path(),
    default=os.path.join(PANQEC_DIR, 'plots'),
    help='Directory to save plots in.'
)
@click.argument(
    'paths', nargs=-1, type=click.Path(exists=True),
)
def analyze(paths, overrides, plot_dir):
    """Analyze the data at given paths."""

    # Use headless plotting and ignore warnings from matplotlib.
    import matplotlib
    matplotlib.use('Agg')
    import warnings
    warnings.filterwarnings('ignore')

    analysis = Analysis(list(paths), overrides=overrides, verbose=True)
    analysis.analyze(progress=tqdm)
    analysis.make_plots(plot_dir)
    analysis.save(os.path.join(plot_dir, 'analysis.json.gz'))


@click.command()
@click.argument('log_file', type=str, required=True)
@click.option(
    '-i', '--interval', default=10, type=click.INT,
    show_default=True
)
def monitor_usage(log_file: str, interval: float = 10):
    """Continously monitor CPU usage by logging to file at intervals.

    Parameters
    ----------
    log_file : str
        Path to log file where messages are saved.
    interval : int
        Interval at which to check usage, in seconds.
    """
    ppid = os.getppid()
    if not os.path.isfile(log_file):
        with open(log_file, 'w') as f:
            f.write(f'Log file for {ppid}\n')
    while True:
        cpu_usage = psutil.cpu_percent(percpu=True)
        mean_cpu_usage = np.mean(cpu_usage)
        n_cores = len(cpu_usage)
        time_now = datetime.datetime.now()
        mem = psutil.virtual_memory()
        ram_usage = mem.percent
        ram_total = mem.total/2**30
        message = (
            f'{time_now} CPU usage {mean_cpu_usage:.2f}% '
            f'({n_cores} cores) '
            f'RAM {ram_usage:.2f}% ({ram_total:.2f} GiB tot)'
        )
        with open(log_file, 'a') as f:
            f.write(message + '\n')
        time.sleep(interval)


@click.command()
@click.option(
    '-d', '--data_dir', required=True, type=str,
    help='Directory to save input .json files, as'
    '`[data_dir]/inputs/input_bias_[eta].json`'
)
@click.option(
    '--decoder_class', default='BeliefPropagationOSDDecoder',
    show_default=True,
    type=click.Choice(list(DECODERS.keys())),
    help='Decoder class name. '
    'Use `panqec ls decoders` to find the list of all decoders.'
)
@click.option(
    '-s', '--sizes', default='3x3,5x5,7x7', type=str,
    show_default=True,
    help='List of sizes, separated by a comma, where each size'
    'has the form [Lx]x[Ly]x[Lz]'
)
@click.option(
    '--bias', default='Z', type=click.Choice(['X', 'Y', 'Z']),
    show_default=True,
    help='Pauli noise bias'
)
@click.option(
    '--eta', default='0.5', type=str,
    show_default=True,
    help='Bias ratio'
)
@click.option(
    '--prob', default='0:0.6:0.005', type=str,
    show_default=True,
    help='min:max:step or single value or list of values'
)
@click.option(
    '--code_class', default=None, type=str,
    show_default=True,
    help='Code class name, e.g. Toric3DCode. '
    'Use `panqec ls codes` to find the list of all codes'
)
@click.option(
    '--noise_class', default='PauliErrorModel', type=str,
    show_default=True,
    help='Error model class name, e.g. PauliErrorModel. '
    'Use `panqec ls error_models` to find the list of all error models'
)
@click.option(
    '--deformation_name', default=None, type=str,
    show_default=True,
    help='Name of the Clifford deformation to use in our noise, e.g. XZZX'
)
@click.option(
    '-m', '--method', default='direct',
    show_default=True,
    type=click.Choice(['direct', 'splitting']),
    help='Simulation method, between "direct" (simple Monte-Carlo simulation)'
    'and "splitting" (Metropolis-Hastings for low error rates)'
)
@click.option(
    '-l', '--label', default=None,
    show_default=True,
    type=str,
    help='Label for the inputs'
)
def generate_input(
    data_dir, sizes, decoder_class, bias, eta, prob,
    code_class, noise_class, deformation_name, method, label
):
    """Generate the json files of every experiment.

    \b
    Example:
    panqec generate-input -i data/toric-3d-code/ \\
            --code_class Toric3DCode \\
            --noise_class PauliErrorModel \\
            -s 3x3x3,5x5x5,7x7x7, --decoder BeliefPropagationOSDDecoder \\
            --bias Z --eta '10,100,1000,inf' \\
            --prob 0:0.5:0.005
    """
    input_dir = os.path.join(data_dir, 'inputs')
    os.makedirs(input_dir, exist_ok=True)

    error_rates = read_range_input(prob)
    bias_ratios = read_bias_ratios(eta)

    for eta in bias_ratios:
        direction = get_direction_from_bias_ratio(bias, eta)

        L_list = [s.split('x') for s in sizes.split(',')]
        code_parameters = [
            {
                "L_x": int(L[0]),
                "L_y": int(L[1]) if len(L) >= 2 else int(L[0]),
                "L_z": int(L[2]) if len(L) == 3 else int(L[0])
            }
            for L in L_list
        ]
        code_dict = {
            "name": code_class,
            "parameters": code_parameters
        }

        noise_parameters = direction
        if deformation_name is not None:
            noise_parameters['deformation_name'] = deformation_name

        error_model_dict = {
            "name": noise_class,
            "parameters": noise_parameters
        }

        if decoder_class == "BeliefPropagationOSDDecoder":
            decoder_parameters = {'max_bp_iter': 1000,
                                  'osd_order': 100}
        else:
            decoder_parameters = {}

        method_parameters = {}
        if method == 'splitting':
            method_parameters['n_init_runs'] = 20000

        method_dict = {
            'name': method,
            'parameters': method_parameters
        }

        decoder_dict = {"name": decoder_class,
                        "parameters": decoder_parameters}

        if label is None:
            label = 'experiment'

        ranges_dict = {"label": label,
                       "method": method_dict,
                       "code": code_dict,
                       "error_model": error_model_dict,
                       "decoder": decoder_dict,
                       "error_rate": error_rates}

        json_dict = {"comments": "",
                     "ranges": ranges_dict}

        filename = os.path.join(input_dir, f'{label}.json')

        with open(filename, 'w') as json_file:
            json.dump(json_dict, json_file, indent=4)


@click.group(invoke_without_command=True)
@click.pass_context
def slurm(ctx):
    """Routines for generating and running slurm scripts."""
    if not ctx.invoked_subcommand:
        print(ctx.get_help())


@click.command()
@click.argument(
    'result-files', type=click.Path(exists=True), nargs=-1, required=True
)
@click.option(
    '-o', '--output_file', type=str, default='merged-results.json.gz',
    show_default=True
)
def merge_results(
    result_files: str,
    output_file: str = 'merged-results.json.gz'
):
    """Merge result directories that had been split into outdir."""

    print(f'Merging {len(result_files)} files to {output_file}')
    combined_results = []
    for file in result_files:
        try:
            combined_results.append(load_json(file))
        except JSONDecodeError:
            print(f'Error reading {file}, skipping')

    save_json(combined_results, output_file)


@click.command()
@click.argument('header-file', required=True)
@click.option('--output-file', '-o', type=str, required=True)
@click.option('-d', '--data-dir', type=click.Path(exists=True), required=True)
@click.option(
    '--cluster', type=click.Choice(['sge', 'slurm', 'pbs']), required=True
)
@click.option('-n', '--n-nodes', type=click.INT, required=True)
@click.option('-w', '--wall-time', type=str, required=True)
@click.option('-m', '--memory', type=str, required=True)
@click.option(
    '-t', '--trials', type=click.INT, show_default=True, required=True
)
@click.option(
    '-c', '--n-cores', default=None, type=click.INT, show_default=True
)
@click.option('-p', '--partition', default='pml', type=str, show_default=True)
@click.option('-q', '--qos', default='dpart', type=str, show_default=True)
@click.option('--working-dir', default='.', type=str, show_default=True)
@click.option(
    '--delete-existing', is_flag=True, default=False, show_default=True
)
def generate_cluster_script(
    header_file: str,
    output_file: str,
    data_dir: str,
    cluster: str,
    n_nodes: int,
    wall_time: str,
    memory: str,
    trials: int,
    n_cores: Optional[int] = None,
    partition: str = 'pml',
    qos: str = 'dpart',
    working_dir: str = '.',
    delete_existing: bool = False
):
    """Generate a generic cluster script from a given header file"""

    with open(header_file, 'r') as f:
        text = f.read()

    inputs_dir = os.path.join(data_dir, 'inputs')
    assert os.path.isdir(inputs_dir), (
        f'{inputs_dir} missing, please create it and generate inputs'
    )
    log_dir = os.path.join(data_dir, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    name = os.path.basename(data_dir)
    data_dir = os.path.abspath(data_dir)
    delete_option = "--delete-existing" if delete_existing else ""

    i_node_dict = {
        'sge': "$SGE_TASK_ID",
        'slurm': "$SLURM_ARRAY_TASK_ID",
        'pbs': "PBS_ARRAY_INDEX"
    }
    job_id_dict = {
        'sge': '${JOB_ID}',
        'slurm': '${SLURM_JOB_ID}',
        'pbs': '$PBS_JOBID'
    }

    i_node = i_node_dict[cluster]
    job_id = job_id_dict[cluster]

    # If n_cores hasn't been specified, take the maximum number of cores
    if n_cores is None:
        n_cores = multiprocessing.cpu_count()

    replace_map = {
        '${TIME}': wall_time,
        '${MEMORY}': memory,
        '${NAME}': name,
        '${N_NODES}': str(n_nodes),
        '${DATA_DIR}': data_dir,
        '${TRIALS}': str(trials),
        '${N_CORES}': str(n_cores),
        '${QUEUE}': partition,
        '${QOS}': qos,
        '${WORKING_DIR}': working_dir
    }
    for template_string, value in replace_map.items():
        text = text.replace(template_string, value)

    monitor_command = "panqec monitor-usage " \
        f"{log_dir}/usage_{job_id}_{i_node}.txt &"
    run_command = "panqec run-parallel " \
        f"-d {data_dir} " \
        f"-n {n_nodes} " \
        f"-j {i_node} " \
        f"-c {n_cores} " \
        f"-t {trials} " \
        f"{delete_option}"

    with open(output_file, 'w') as f:
        f.write(text + "\n\n")
        f.write(monitor_command + "\n\n")
        f.write(run_command + "\n\n")
        f.write("date")

    print(f'Wrote to {output_file}')


@click.command()
@click.argument('folder', required=True, type=click.Choice(
    ['all', 'out', 'sbatch'],
    case_sensitive=False
))
def clear(folder):
    """Clear generated files."""
    if folder == 'out' or folder == 'all':
        clear_out_folder()
    if folder == 'sbatch' or folder == 'all':
        clear_sbatch_folder()


@click.command()
@click.argument('name', required=True)
def count(name):
    """Count number of input parameters contained."""
    n_runs = count_input_runs(name)
    print(n_runs)


@click.command()
def status():
    """Show the status of running jobs."""
    get_status()


@click.command
@click.argument(
    'data_dirs', type=click.Path(exists=True), nargs=-1,
    required=True
)
def check_usage(data_dirs: str):
    """Check usage of resources."""
    log_dirs = []
    if data_dirs:
        for data_dir in data_dirs:
            log_dir = os.path.join(data_dir, 'logs')
            if not os.path.isdir(log_dir):
                print(f'{log_dir} not a directory')
            else:
                log_dirs.append(log_dir)

    summarize_usage(log_dirs)


@click.command
@click.argument(
    'log_dir', type=click.Path(exists=True), required=True
)
@click.option(
    '-a', '--show-all', is_flag=True, default=False,
    help='Show progress on all the cores individually'
)
def check_progress(log_dir: str, show_all: bool = False):
    """Check usage of resources."""
    if not os.path.isdir(log_dir):
        print(f'{log_dir} not a directory')

    if 'progress' in os.listdir(log_dir):
        progress_dir = os.path.join(log_dir, 'progress')
    else:
        progress_dir = log_dir

    list_files = glob(os.path.join(progress_dir, 'progress_*.txt'))

    total_n = 0
    total_k = 0

    for filename in list_files:
        with open(filename, "r") as f:
            k, n = f.read().split("/")

            total_n += int(n)
            total_k += int(k)

        if show_all:
            progress_bar(int(k), int(n))

    if len(list_files) > 0:
        if show_all:
            print("\n")
        print("Total progress:\n")
        progress_bar(total_k, total_n)


slurm.add_command(status)
slurm.add_command(count)
slurm.add_command(clear)
cli.add_command(start_gui)
cli.add_command(run)
cli.add_command(run_parallel)
cli.add_command(ls)
cli.add_command(slurm)
cli.add_command(generate_input)
cli.add_command(monitor_usage)
cli.add_command(merge_results)
cli.add_command(generate_cluster_script)
cli.add_command(check_usage)
cli.add_command(check_progress)
cli.add_command(analyze)
