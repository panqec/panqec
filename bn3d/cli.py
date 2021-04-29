from typing import Optional
import click
import bn3d
from .config import codes, error_models, decoders


@click.group()
@click.version_option(version=bn3d.__version__, prog_name='bn3d')
def cli():
    """
    bn3d - biased noise in 3D simulations.

    See bn3d COMMAND --help for command-specific help.
    """
    pass


@click.command()
@click.option('-f', '--file', 'file_', type=click.File())
@click.option('-c', '--code')
@click.option('-n', '--noise')
@click.option('-d', '--decoder')
@click.option('-p', '--probability', type=click.FLOAT)
@click.option('-t', '--trials', type=click.INT)
def run(
    file_: Optional[str] = None,
    code: str = 'ToricCode3D(3, 3, 3)',
    noise: str = 'PauliErrorModel(1, 0, 0)',
    decoder: str = 'PyMatchingSweepDecoder3D()',
    probability: float = 0.5,
    trials: int = 10
):
    if file_ is not None:
        raise NotImplementedError('Run file not working yet')
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
            '    ' + name for name in sorted(codes.keys())
        ]))
    if model_type is None or model_type == 'noise':
        print('Error Models (Noise):')
        print('\n'.join([
            '    ' + name for name in sorted(error_models.keys())
        ]))
    if model_type is None or model_type == 'decoders':
        print('Decoders:')
        print('\n'.join([
            '    ' + name for name in sorted(decoders.keys())
        ]))


cli.add_command(run)
cli.add_command(ls)
