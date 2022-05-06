"""
Settings from environmental variables and config files.

:Author:
    Eric Huang
"""
import os
from dotenv import load_dotenv
from .codes import (
    Toric3DCode, Toric2DCode,
    RotatedPlanar3DCode, XCubeCode,
    RotatedToric3DCode, RhombicCode
)
from .decoders import (
    Toric3DPymatchingDecoder, SweepMatchDecoder,
    RotatedSweepMatchDecoder, RotatedInfiniteZBiasDecoder
)
from .decoders.bposd.bposd_decoder import BeliefPropagationOSDDecoder
from .decoders.sweepmatch._toric_2d_match_decoder import Toric2DPymatchingDecoder
from .error_models import (
    DeformedXZZXErrorModel, DeformedXYErrorModel,
    DeformedRhombicErrorModel, DeformedRandomErrorModel
)
from .decoders import (
    DeformedSweepMatchDecoder, FoliatedMatchingDecoder,
    DeformedRotatedSweepMatchDecoder
)
from .error_models import PauliErrorModel

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Load the .env file into environmental variables.
if os.getenv('PANQEC_DIR') is None:
    load_dotenv()

PANQEC_DARK_THEME = False
if os.getenv('PANQEC_DARK_THEME'):
    PANQEC_DARK_THEME = bool(os.getenv('PANQEC_DARK_THEME'))

# Fallback is to use temp dir inside repo if PANQEC_DIR is not available.
PANQEC_DIR = os.path.join(
    os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
    'temp'
)

# Load the output directory from environmental variables.
if os.getenv('PANQEC_DIR') is not None:
    PANQEC_DIR = os.path.abspath(str(os.getenv('PANQEC_DIR')))
    if not os.path.isdir(PANQEC_DIR):
        raise FileNotFoundError(
            f'PANQEC_DIR={PANQEC_DIR} is not a valid directory. '
            'Check .env configuration.'
        )

# Register your models here.
CODES = {
    'Toric2DCode': Toric2DCode,
    'Toric3DCode': Toric3DCode,
    'RhombicCode': RhombicCode,
    'RotatedPlanar3DCode': RotatedPlanar3DCode,
    'RotatedToric3DCode': RotatedToric3DCode,
    'XCubeCode': XCubeCode
}
ERROR_MODELS = {
    'PauliErrorModel': PauliErrorModel,
    'DeformedXZZXErrorModel': DeformedXZZXErrorModel,
    'DeformedRandomErrorModel': DeformedRandomErrorModel,
    'DeformedXYErrorModel': DeformedXYErrorModel,
    'DeformedRhombicErrorModel': DeformedRhombicErrorModel,
}
DECODERS = {
    'Toric2DPymatchingDecoder': Toric2DPymatchingDecoder,
    'Toric3DPymatchingDecoder': Toric3DPymatchingDecoder,
    'SweepMatchDecoder': SweepMatchDecoder,
    'RotatedSweepMatchDecoder': RotatedSweepMatchDecoder,
    'DeformedSweepMatchDecoder': DeformedSweepMatchDecoder,
    'FoliatedMatchingDecoder': FoliatedMatchingDecoder,
    'DeformedRotatedSweepMatchDecoder': DeformedRotatedSweepMatchDecoder,
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
