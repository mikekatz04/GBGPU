from .globals import get_backend, get_first_backend, get_logger
from ..cutils import Backend
from typing import Optional, Sequence, TypeVar, Union
import types


from gpubackendtools import ParallelModuleBase


class GBGPUParallelModule(ParallelModuleBase):
    def __init__(self, force_backend=None):
        force_backend_in = ('gbgpu', force_backend) if isinstance(force_backend, str) else force_backend
        super().__init__(force_backend_in)
