import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parent / 'jepa'))

from .encoders import VJEPABase,VJEPALarge
