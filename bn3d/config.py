"""
Settings from environmental variables and config files.

:Author:
    Eric Huang
"""
import os
from dotenv import load_dotenv
from qecsim.models.basic import FiveQubitCode
from qecsim.models.toric import ToricCode
from qecsim.models.generic import NaiveDecoder
from .tc3d import (
    ToricCode3D, Toric3DPymatchingDecoder, SweepMatchDecoder,
    RotatedPlanarCode3D, RotatedToricCode3D, RotatedSweepMatchDecoder,
    RotatedInfiniteZBiasDecoder
)
from .rhombic import RhombicCode
from .bp_os_decoder import BeliefPropagationOSDDecoder
from .tc2d import Toric2DPymatchingDecoder
from .deform import (
    DeformedXZZXErrorModel, DeformedXYErrorModel,
    DeformedSweepMatchDecoder, DeformedRhombicErrorModel,
    FoliatedMatchingDecoder, DeformedToric3DPymatchingDecoder,
)
from .noise import PauliErrorModel, XNoiseOnYZEdgesOnly

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Load the .env file into environmental variables.
if os.getenv('BN3D_DIR') is None:
    load_dotenv()

BN3D_DARK_THEME = False
if os.getenv('BN3D_DARK_THEME'):
    BN3D_DARK_THEME = bool(os.getenv('BN3D_DARK_THEME'))

# Fallback is to use temp dir inside repo if BN3D_DIR is not available.
BN3D_DIR = os.path.join(
    os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
    'temp'
)

# Load the output directory from environmental variables.
if os.getenv('BN3D_DIR') is not None:
    BN3D_DIR = os.path.abspath(str(os.getenv('BN3D_DIR')))
    if not os.path.isdir(BN3D_DIR):
        raise FileNotFoundError(
            f'BN3D_DIR={BN3D_DIR} is not a valid directory. '
            'Check .env configuration.'
        )

# Register your models here.
CODES = {
    'ToricCode': ToricCode,
    'ToricCode3D': ToricCode3D,
    'RhombicCode': RhombicCode,
    'RotatedPlanarCode3D': RotatedPlanarCode3D,
    'RotatedToricCode3D': RotatedToricCode3D,
    'FiveQubitCode': FiveQubitCode,
}
ERROR_MODELS = {
    'PauliErrorModel': PauliErrorModel,
    'DeformedXZZXErrorModel': DeformedXZZXErrorModel,
    'DeformedXYErrorModel': DeformedXYErrorModel,
    'DeformedRhombicErrorModel': DeformedRhombicErrorModel,
    'XNoiseOnYZEdgesOnly': XNoiseOnYZEdgesOnly,
}
DECODERS = {
    'Toric2DPymatchingDecoder': Toric2DPymatchingDecoder,
    'Toric3DPymatchingDecoder': Toric3DPymatchingDecoder,
    'SweepMatchDecoder': SweepMatchDecoder,
    'RotatedSweepMatchDecoder': RotatedSweepMatchDecoder,
    'NaiveDecoder': NaiveDecoder,
    'DeformedSweepMatchDecoder': DeformedSweepMatchDecoder,
    'FoliatedMatchingDecoder': FoliatedMatchingDecoder,
    'DeformedToric3DPymatchingDecoder': DeformedToric3DPymatchingDecoder,
    'BeliefPropagationOSDDecoder': BeliefPropagationOSDDecoder,
    'RotatedInfiniteZBiasDecoder': RotatedInfiniteZBiasDecoder
}

# Slurm automation config.
SLURM_DIR = os.path.join(os.path.dirname(BASE_DIR), 'slurm')
if os.getenv('SLURM_DIR') is not None:
    SLURM_DIR = os.path.abspath(str(os.getenv('SLURM_DIR')))

SBATCH_TEMPLATE = os.path.join(
    os.path.dirname(BASE_DIR), 'scripts', 'template.sbatch'
)
NIST_TEMPLATE = os.path.join(
    os.path.dirname(BASE_DIR), 'scripts', 'nist.sbatch'
)

# Slurm username for reporting status.
SLURM_USERNAME = None
if os.getenv('USER') is not None:
    SLURM_USERNAME = os.getenv('USER')
elif os.getenv('USERNAME') is not None:
    SLURM_USERNAME = os.getenv('USERNAME')
