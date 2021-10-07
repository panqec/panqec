from multiprocessing import Pool, cpu_count
import click
from .controllers import DataManager
from .core import start_sampling, generate_inputs
from .config import SPIN_MODELS, DISORDER_MODELS


@click.group(invoke_without_command=True)
@click.pass_context
def statmech(ctx):
    """Routines for doing stat mech MCMC runs."""
    if not ctx.invoked_subcommand:
        print(ctx.get_help())


@click.command()
@click.argument('data_dir', required=True)
def status(data_dir):
    """Show the status of running jobs."""
    print(f'Showing statmech status in {data_dir}')

    dm = DataManager(data_dir)
    for subdir in dm.subdirs.keys():
        print('{}: {} files'.format(subdir, dm.count(subdir)))


@click.command()
@click.argument('data_dir', required=True)
def sample(data_dir):
    """Perform MCMC runs.."""
    n_workers = cpu_count()
    arguments = [
        data_dir
        for i_worker in range(n_workers)
    ]
    print(f'Sampling over {n_workers} CPUs')
    with Pool() as pool:
        pool.map(start_sampling, arguments)


@click.command()
@click.argument('data_dir', required=True)
def generate(data_dir):
    """Generate inputs for MCMC."""
    generate_inputs(data_dir)


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
