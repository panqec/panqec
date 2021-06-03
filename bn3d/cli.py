import os
from typing import Optional
import click
import bn3d
from tqdm import tqdm
from .app import run_file
from .config import CODES, ERROR_MODELS, DECODERS
from .slurm import generate_sbatch, get_status, generate_sbatch_nist


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
def run(
    ctx,
    file_: Optional[str],
    trials: int
):
    """Run a single job or run many jobs from input file."""
    if file_ is not None:
        run_file(os.path.abspath(file_), trials, progress=tqdm)
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
@click.option('--n_trials', default=1000, type=click.INT, show_default=True)
@click.option('--nodes', default=1, type=click.INT, show_default=True)
@click.option('--ntasks', default=1, type=click.INT, show_default=True)
@click.option('--cpus_per_task', default=40, type=click.INT, show_default=True)
@click.option('--mem', default=10000, type=click.INT, show_default=True)
@click.option('--time', default='10:00:00', show_default=True)
def gennist(n_trials, nodes, ntasks, cpus_per_task, mem, time):
    """Generate sbatch files for NIST cluster."""
    generate_sbatch_nist(
        n_trials, nodes, ntasks, cpus_per_task, mem, time
    )


@click.command()
def status():
    """Show the status of running jobs."""
    get_status()


slurm.add_command(gen)
slurm.add_command(gennist)
slurm.add_command(status)
cli.add_command(run)
cli.add_command(ls)
cli.add_command(slurm)
