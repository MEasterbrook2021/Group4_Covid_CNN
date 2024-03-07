from .model_custom import *
from .model_resnet import *

from enum import Enum

class ModelTypes(Enum):
    RESNET = "resnet"
    CUSTOM = "custom"