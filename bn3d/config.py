"""
Settings from environmental variables and config files.

:Author:
    Eric Huang
"""
import os
from dotenv import load_dotenv
from qecsim.models.basic import FiveQubitCode
from qecsim.models.generic import NaiveDecoder
from .tc3d import ToricCode3D, Toric3DPymatchingDecoder, SweepMatchDecoder
from .deform import DeformedPauliErrorModel, DeformedSweepMatchDecoder
from .noise import PauliErrorModel

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
    'ToricCode3D': ToricCode3D,
    'FiveQubitCode': FiveQubitCode,
}
ERROR_MODELS = {
    'PauliErrorModel': PauliErrorModel,
    'DeformedPauliErrorModel': DeformedPauliErrorModel,
}
DECODERS = {
    'Toric3DPymatchingDecoder': Toric3DPymatchingDecoder,
    'SweepMatchDecoder': SweepMatchDecoder,
    'NaiveDecoder': NaiveDecoder,
    'DeformedSweepMatchDecoder': DeformedSweepMatchDecoder,
}
