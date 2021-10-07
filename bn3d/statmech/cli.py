import click
from .controllers import DataManager


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
def sample(data_dir, max_tau):
    """Perform MCMC runs.."""


@click.command()
@click.argument('data_dir', required=True)
def generate(data_dir):
    """Generate inputs for MCMC."""


@click.command()
def models():
    """List available models."""


statmech.add_command(status)
statmech.add_command(sample)
statmech.add_command(generate)
statmech.add_command(models)
