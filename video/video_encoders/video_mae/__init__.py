import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parent / 'ClinicalMAE'))
print(sys.path[-1])

from .ClinicalMAE.encoders import VideoMAEv2Base
