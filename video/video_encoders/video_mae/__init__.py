import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parent / 'ClinicalMAE'))

from .ClinicalMAE.encoders import VideoMAEv2Base
