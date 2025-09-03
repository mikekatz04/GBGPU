from .globals import get_backend, get_first_backend, get_logger
from ..cutils import Backend
from typing import Optional, Sequence, TypeVar, Union
import types

class ParallelModuleBase:  # (Citable):
    """
    Base class for modules that can use a GPU (or revert back to CPU).

    This class mainly handles backend selection. Each backend offers accelerated
    computations on a specific device (cpu, CUDA 11.x enabled GPU, CUDA 12.x enabled GPU).

    args:
        force_backend (str, optional): Name of the backend to use
    """

    _backend_name: str

    def __init__(self, /, force_backend = None):
        if force_backend is not None:
            if isinstance(force_backend, Backend):
                force_backend = force_backend.name
            self._backend_name = get_backend(force_backend).name
        else:
            self._backend_name = get_first_backend(self.supported_backends()).name

    @property
    def backend(self) -> Backend:
        """Access the underlying backend."""
        return get_backend(self._backend_name)

    @classmethod
    def supported_backends(cls) -> Sequence[str]:
        """List of backends supported by a parallel module by order of preference."""
        raise NotImplementedError(
            "Class {} does not implement the supported_backends method.".format(cls)
        )

    @staticmethod
    def CPU_ONLY() -> list[str]:
        """List of supported backend for CPU only class"""
        return ["cpu"]

    @staticmethod
    def GPU_RECOMMENDED() -> list[str]:
        """List of supported backends for GPU-recommended class with CPU support"""
        return ["cuda12x", "cuda11x", "cpu"]

    @staticmethod
    def CPU_RECOMMENDED_WITH_GPU_SUPPORT() -> list[str]:
        """List of supported backends for CPU-recommended class with GPU support"""
        return ["cpu", "cuda12x", "cuda11x"]

    @staticmethod
    def GPU_ONLY() -> list[str]:
        """List of supported backends for GPU-only class"""
        return ["cuda12x", "cuda11x"]

    @property
    def xp(self) -> types.ModuleType:
        """Return the module providing ndarray capabilities"""
        return self.backend.xp

    @property
    def backend_name(self) -> str:
        """Return the name of current backend"""
        return self.backend.name

    ParallelModuleDerivate = TypeVar(
        "ParallelModuleDerivate", bound="ParallelModuleBase"
    )

    def build_with_same_backend(
        self,
        module_class: type[ParallelModuleDerivate],
        args: Optional[list] = None,
        kwargs: Optional[dict] = None,
    ) -> ParallelModuleDerivate:
        """
        Build an instance of `module_class` with same backend as current object.

        args:
          module_class: class of the object to be built, must derive from ParallelModuleBase
          args (list, optional): positional arguments for module_class constructor
          kwargs (dict, optional): keyword arguments for module_class constructor
                                   (the 'force_backend' argument will be ignored and replaced)
        """
        args = [] if args is None else args
        return module_class(*args, **self.adapt_backend_kwargs(kwargs=kwargs))

    def adapt_backend_kwargs(self, kwargs: Optional[dict] = None) -> dict:
        """Adapt a set of keyword arguments to add/set 'force_backend' to current backend"""
        if kwargs is None:
            kwargs = {}
        kwargs["force_backend"] = self.backend_name
        return kwargs
