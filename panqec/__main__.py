"""
Command line interface.
"""

from multiprocessing import freeze_support
from .cli import cli


if __name__ == '__main__':
    freeze_support()
    cli(prog_name="python -m panqec")
