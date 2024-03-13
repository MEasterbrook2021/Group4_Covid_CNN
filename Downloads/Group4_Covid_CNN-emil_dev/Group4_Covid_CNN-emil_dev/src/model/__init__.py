from .model_custom import *
from .model_resnet import *

from enum import Enum

class ModelDir:
    PATH = Path("models")
    PATH_OUTPUT = PATH / Covidx_CXR2.NAME

class ModelTypes(Enum):
    RESNET = "resnet"
    CUSTOM = "custom"