import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent / "ClinicalMAE"))

from .encoders import VideoMAEv2Base
