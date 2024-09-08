import getpass
import os
import sys

__USERNAME = getpass.getuser()
_BASE_DIR = f'/U_20240603_ZSH_SMIL/emerge/EmergencyIndex/'
DATA_FOLDER = os.path.join('/U_20240603_ZSH_SMIL/emerge/EmergencyIndex/data', 'datasets')
GENERATION_FOLDER = os.path.join(_BASE_DIR, 'output')
os.makedirs(GENERATION_FOLDER, exist_ok=True)

