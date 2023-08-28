"""
https://docs.cupy.dev/en/stable/reference/environment.html
https://docs.cupy.dev/en/stable/user_guide/performance.html
import os
os.environ['CUPY_TF32']='1' # only available on GPUs with compute capability 8.0 or higher
os.environ['CUPY_ACCELERATORS']='cub'
"""
import configparser

__version__ = '1.3.2'

from .xparray import *
from .tensor import *
from .VariableFunctions import *
from .serialization import *
from . import nn
from . import optim
