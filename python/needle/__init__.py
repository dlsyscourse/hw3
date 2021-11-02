from .autograd import Tensor
from . import ops
from .ops import *

from . import numpy_backend
from . import nd_backend

from .nd_backend import cuda, cpu
from .numpy_backend import numpy_device
