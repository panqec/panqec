from dotenv import load_dotenv
import os

# Load the .env file into environmental variables.
load_dotenv()

# Load the output directory from environmental variables.
BN3D_DIR = os.getenv('BN3D_DIR')

# Fallback is to use temp dir inside repo if BN3D_DIR is not available.
if BN3D_DIR is None or not os.path.exists(BN3D_DIR):
    BN3D_DIR = os.path.join(
        os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
        'temp'
    )
