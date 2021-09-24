import os
from typing import Optional
import click
import bn3d
from tqdm import tqdm
import numpy as np
import json
from .app import run_file
from .config import CODES, ERROR_MODELS, DECODERS, BN3D_DIR
from .slurm import (
    generate_sbatch, get_status, generate_sbatch_nist, count_input_runs,
    clear_out_folder, clear_sbatch_folder
)
from bn3d.plots._hashing_bound import (
    generate_points_triangle
)


@click.group(invoke_without_command=True)
@click.version_option(version=bn3d.__version__, prog_name='bn3d')
@click.pass_context
def cli(ctx):
    """
    bn3d - biased noise in 3D simulations.

    See bn3d COMMAND --help for command-specific help.
    """
    if not ctx.invoked_subcommand:
        print(ctx.get_help())


@click.command()
@click.pass_context
@click.option('-f', '--file', 'file_')
@click.option('-t', '--trials', default=100, type=click.INT, show_default=True)
@click.option('-s', '--start', default=None, type=click.INT, show_default=True)
@click.option(
    '-o', '--output_dir', default=BN3D_DIR, type=click.STRING,
    show_default=True
)
@click.option(
    '-n', '--n_runs', default=None, type=click.INT, show_default=True
)
def run(
    ctx,
    file_: Optional[str],
    trials: int,
    start: Optional[int],
    n_runs: Optional[int],
    output_dir: Optional[str]
):
    """Run a single job or run many jobs from input file."""
    if file_ is not None:
        run_file(
            os.path.abspath(file_), trials,
            start=start, n_runs=n_runs, progress=tqdm,
            output_dir=output_dir
        )
    else:
        print(ctx.get_help())


@click.command()
@click.argument('model_type', required=False, type=click.Choice(
    ['codes', 'noise', 'decoders'],
    case_sensitive=False
))
def ls(model_type=None):
    """List available codes, noise models and decoders."""
    if model_type is None or model_type == 'codes':
        print('Codes:')
        print('\n'.join([
            '    ' + name for name in sorted(CODES.keys())
        ]))
    if model_type is None or model_type == 'noise':
        print('Error Models (Noise):')
        print('\n'.join([
            '    ' + name for name in sorted(ERROR_MODELS.keys())
        ]))
    if model_type is None or model_type == 'decoders':
        print('Decoders:')
        print('\n'.join([
            '    ' + name for name in sorted(DECODERS.keys())
        ]))


@click.command()
@click.option('-i', '--input_dir', required=True, type=str)
@click.option('-d', '--decoder', required=True, type=click.Choice(
    ['bp-osd', 'bp-osd-2', 'sweepmatch'],
    case_sensitive=False
))
def generate_input(input_dir, decoder):
    """Generate the json files of every experiments"""

    codes = ["cubic", "rhombic"]
    delta = 0.005
    probabilities = np.arange(0, 0.5+delta, delta).tolist()
    directions = generate_points_triangle()
    for code in codes:
        for deformed in [True, False]:
            for direction in directions:
                for p in probabilities:
                    label = "deformed" if deformed else "regular"
                    label += f"-{code}"
                    label += f"-{decoder}"
                    label += (
                        f"-{direction['r_x']:.2f}-{direction['r_y']:.2f}"
                        f"-{direction['r_z']:.2f}"
                    )
                    label += f"-p-{p:.3f}"

                    code_model = (
                        "ToricCode3D" if code == 'cubic' else "RhombicCode"
                    )
                    code_parameters = [
                        {"L_x": 4},
                        {"L_x": 6},
                        {"L_x": 8},
                        {"L_x": 10},
                    ]
                    code_dict = {
                        "model": code_model,
                        "parameters": code_parameters
                    }

                    noise_model = "Deformed" if deformed else ""
                    noise_model += "PauliErrorModel"
                    noise_parameters = direction
                    noise_dict = {
                        "model": noise_model,
                        "parameters": noise_parameters
                    }

                    if decoder == "sweepmatch":
                        decoder_model = "SweepMatchDecoder"
                        if deformed:
                            decoder_model = "Deformed" + decoder_model
                        decoder_dict = {"model": decoder_model}
                    elif decoder == "bp-osd":
                        decoder_model = "BeliefPropagationOSDDecoder"
                        decoder_parameters = {'deformed': deformed,
                                              'max_bp_iter': 10}
                        decoder_dict = {"model": decoder_model,
                                        "parameters": decoder_parameters}
                    elif decoder == "bp-osd-2":
                        decoder_model = "BeliefPropagationOSDDecoder"
                        decoder_parameters = {'deformed': deformed,
                                              'joschka': True,
                                              'max_bp_iter': 10}
                        decoder_dict = {"model": decoder_model,
                                        "parameters": decoder_parameters}
                    else:
                        raise ValueError("Decoder not recognized")

                    ranges_dict = {"label": label,
                                   "code": code_dict,
                                   "noise": noise_dict,
                                   "decoder": decoder_dict,
                                   "probability": [p]}

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
@click.option('--n_trials', default=1000, type=click.INT, show_default=True)
@click.option('--partition', default='defq', show_default=True)
@click.option('--time', default='10:00:00', show_default=True)
@click.option('--cores', default=1, type=click.INT, show_default=True)
def gen(n_trials, partition, time, cores):
    """Generate sbatch files."""
    generate_sbatch(n_trials, partition, time, cores)


@click.command()
@click.argument('name', required=True)
@click.option('--n_trials', default=1000, type=click.INT, show_default=True)
@click.option('--nodes', default=1, type=click.INT, show_default=True)
@click.option('--ntasks', default=1, type=click.INT, show_default=True)
@click.option('--cpus_per_task', default=40, type=click.INT, show_default=True)
@click.option('--mem', default=10000, type=click.INT, show_default=True)
@click.option('--time', default='10:00:00', show_default=True)
@click.option('--split', default=1, type=click.INT, show_default=True)
@click.option('--partition', default='pml', show_default=True)
@click.option(
    '--cluster', default='nist', show_default=True,
    type=click.Choice(['nist', 'symmetry'])
)
def gennist(
    name, n_trials, nodes, ntasks, cpus_per_task, mem, time, split, partition,
    cluster
):
    """Generate sbatch files for NIST cluster."""
    generate_sbatch_nist(
        name, n_trials, nodes, ntasks, cpus_per_task, mem, time, split,
        partition, cluster
    )


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


slurm.add_command(gen)
slurm.add_command(gennist)
slurm.add_command(status)
slurm.add_command(count)
slurm.add_command(clear)
cli.add_command(run)
cli.add_command(ls)
cli.add_command(slurm)
cli.add_command(generate_input)
