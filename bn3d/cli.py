import os
from typing import Optional, List
import click
import bn3d
from tqdm import tqdm
from .app import run_file
from .config import CODES, ERROR_MODELS, DECODERS
from .slurm import generate_sbatch


@click.group()
@click.version_option(version=bn3d.__version__, prog_name='bn3d')
def cli():
    """
    bn3d - biased noise in 3D simulations.

    See bn3d COMMAND --help for command-specific help.
    """
    pass


@click.command()
@click.option('-f', '--file', 'file_')
@click.option('-t', '--trials', default=100, type=click.INT, show_default=True)
def run(
    file_: Optional[str],
    trials: int
):
    """Run a single job or run many jobs from input file."""
    if file_ is not None:
        run_file(os.path.abspath(file_), trials, progress=tqdm)
    else:
        raise NotImplementedError('Run not working yet')


@click.command()
@click.argument('model_type', required=False, type=click.Choice(
    ['codes', 'noise', 'decoders'],
    case_sensitive=False
))
def ls(model_type=None):
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


@click.group()
def slurm():
    """Routines for producing and running slurm scripts."""
    pass


@click.command()
@click.option('--n_trials', default=1000, type=click.INT, show_default=True)
@click.option('--partition', default='defq', show_default=True)
@click.option('--time', default='10:00:00', show_default=True)
def gen(n_trials, partition, time):
    """Generate sbatch files."""
    generate_sbatch(n_trials, partition, time)


slurm.add_command(gen)
cli.add_command(run)
cli.add_command(ls)
cli.add_command(slurm)
