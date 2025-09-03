from __future__ import annotations
import dataclasses
import enum
import types
import typing
import abc
from typing import Optional, Sequence, TypeVar, Union
from ..utils.exceptions import GBGPUException


class BackendUnavailableException(GBGPUException):
    """Exception raised when the backend is not available."""


class BackendNotInstalled(BackendUnavailableException):
    """Exception raised when the backend has not been installed"""


class MissingDependencies(BackendUnavailableException):
    """Exception raised when the backend has missing dependencies"""

    pip_deps: typing.List[str]
    """List of missing dependencies to install in pip-managed environments"""

    conda_deps: typing.List[str]
    """List of missing dependencies to install in conda-managed environments."""

    def __init__(
        self, *args, pip_deps: typing.List[str], conda_deps: typing.List[str], **kwargs
    ):
        self.pip_deps = pip_deps
        self.conda_deps = conda_deps
        super().__init__(*args, **kwargs)

    def __str__(self) -> str:
        message = super().__str__()
        if self.pip_deps:
            message += """
    If you are using gbgpu in an environment managed using pip, run:
        $ pip install {}

""".format(", ".join(self.pip_deps))
        if self.conda_deps:
            message += """
    If you are using gbgpu in an environment managed using conda, run:
        $ conda install {}

""".format(", ".join(self.conda_deps))
        return message


class MissingDriver(BackendUnavailableException):
    """Exception raised when backend needs driver-like software to be installed."""


class SoftwareException(BackendUnavailableException):
    """Exception raised due to unexpected software error when loading the backend"""


class MissingHardware(BackendUnavailableException):
    """Exception raised when backend needs unavailable hardware."""


@dataclasses.dataclass
class BackendMethods:
    xp: types.ModuleType

@dataclasses.dataclass
class GBGPUBackendMethods(BackendMethods):
    get_ll: typing.Callable[(...), None]
    fill_global: typing.Callable[(...), None]


class Backend:
    """Abstract definition of a backend"""

    name: str
    """Backend unique name"""


    xp: types.ModuleType
    """Reference to package handling the backend ndarrays (numpy or cupy for now)"""

    class Feature(enum.Flag):
        NUMPY = enum.auto()
        """Flag indicating that backend uses numpy.ndarray"""

        CUPY = enum.auto()
        """Flag indicating that backend uses cupy.ndarray"""

        CUDA = enum.auto()
        """Flag indicating that backend uses CUDA devices and offers a set_cuda_device() method."""

        GPU = enum.auto()
        """Flag indicating that backend uses GPU hardware"""

        NONE = 0
        """Special flag representing no activated feature"""

    features: Feature
    """List of Backend features used by a backend"""

    def __init__(self, name: str, methods: BackendMethods, features: Feature):
        self.name = name
        self.xp = methods.xp
        self.features = features

    @staticmethod
    def _check_module_installed(backend_name: str, module_name: str):
        """Check that the module containing the backend implementation is installed."""
        import importlib

        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            raise BackendNotInstalled(
                "The '{}' backend is not installed.".format(backend_name)
            ) from e

    def supports(self, feature: Feature) -> bool:
        """Check whether a backend supports a given feature"""
        return feature in self.features

    @property
    def uses_gpu(self) -> bool:
        """Shortcut to check if a backend supports GPU"""
        return self.supports(feature=Backend.Feature.GPU)

    @property
    def uses_numpy(self) -> bool:
        """Shortcut to check if a backend makes use of NumPy"""
        return self.supports(feature=Backend.Feature.NUMPY)

    @property
    def uses_cupy(self) -> bool:
        """Shortcut to check if a backend makes use of CuPy"""
        return self.supports(feature=Backend.Feature.CUPY)

    @property
    def uses_cuda(self) -> bool:
        """Shortcut to check whether a backend uses CUDA devices"""
        return self.supports(feature=Backend.Feature.CUDA)
    
class GBGPUBackend:
    get_ll: typing.Callable[(...), None]
    fill_global: typing.Callable[(...), None]

    def __init__(self, gbgpu_backend_methods):

        # set direct gbgpu methods
        # pass rest to general backend
        assert isinstance(gbgpu_backend_methods, GBGPUBackendMethods)

        self.get_ll = gbgpu_backend_methods.get_ll
        self.fill_global = gbgpu_backend_methods.fill_global


class CpuBackend(Backend, abc.ABC):
    """Implementation of the CPU backend"""

    backend_name : str = None

    @staticmethod
    @abc.abstractmethod
    def cpu_methods_loader() -> BackendMethods:
        raise not NotImplementedError
    
    @staticmethod
    def check_numpy() -> typing.ModuleType:
        try:
            import numpy
        except (ModuleNotFoundError, ImportError) as e:
            raise MissingDependencies(
                "'cpu' backend requires numpy", pip_deps=["numpy"], conda_deps=["numpy"]
            ) from e
        
        return numpy

    def __init__(self, ):
        """Initialize the CPU backend"""
        name = "cpu"
        if self.backend_name is None:
            raise ValueError("Child class must declare `backend_name` class attribute.")
        
        self._check_module_installed(name, self.backend_name)

        Backend.__init__(
            self,
            name=name,
            methods=self.cpu_methods_loader(),
            features=Backend.Feature.NUMPY,
        )


class GBGPUCpuBackend(CpuBackend, GBGPUBackend):
    """Implementation of the CPU backend"""
    
    backend_name = "gbgpu_backend_cpu"

    def __init__(self, *args, **kwargs):
        CpuBackend.__init__(self, *args, **kwargs)
        GBGPUBackend.__init__(self, self.cpu_methods_loader())

    @staticmethod
    def cpu_methods_loader() -> GBGPUBackendMethods:
        try:
            import gbgpu_backend_cpu.utils
            
        except (ModuleNotFoundError, ImportError) as e:
            raise BackendUnavailableException(
                "'cpu' backend could not be imported."
            ) from e

        numpy = GBGPUCpuBackend.check_numpy()

        return GBGPUBackendMethods(
            get_ll=gbgpu_backend_cpu.utils.get_ll,
            fill_global=gbgpu_backend_cpu.utils.fill_global,
            xp=numpy,
        )


class _CudaBackend(Backend):
    """Implementation of generic CUDA backend"""

    @staticmethod
    def _get_cuda_version() -> typing.Tuple[int, int]:
        """Get the CUDA version or raise an exception"""
        try:
            import pynvml

            pynvml.nvmlInit()
            cuda_version = pynvml.nvmlSystemGetCudaDriverVersion_v2()
        except pynvml.NVMLError_DriverNotLoaded as e:
            raise MissingDriver(
                "CUDA Driver is missing. Ensure you installed it properly."
            ) from e
        except pynvml.NVMLError as e:
            raise SoftwareException(
                "CUDA driver exception: could not detect a CUDA version."
            ) from e
        except AttributeError as e:
            raise SoftwareException(
                "The NVIDIA Management Library (libnvml) does not support the expected method to detect CUDA version."
            ) from e

        cuda_major = cuda_version // 1000
        cuda_minor = (cuda_version % 1000) // 10
        return cuda_major, cuda_minor

    @staticmethod
    def _check_cupy_works(cuda_major: int):
        try:
            import cupy
            import cupy_backends.cuda
        except ImportError as e:
            raise MissingDependencies(
                "CuPy is missing.",
                pip_deps=["cupy-cuda{}x".format(cuda_major)],
                conda_deps=[],
            ) from e

        try:
            cupy_cuda_ver = cupy.cuda.get_local_runtime_version() // 1000
        except RuntimeError as e:
            raise SoftwareException("CuPy could not detect runtime version") from e

        if cupy_cuda_ver != cuda_major:
            raise MissingDependencies(
                "CuPy is installed but supports CUDA {} instead of {}".format(
                    cupy_cuda_ver, cuda_major
                ),
                pip_deps=["cupy-cuda{}x".format(cuda_major)],
                conda_deps=[],
            )

        try:
            _ = cupy.arange(1)
            _ = cupy.cuda.runtime.getDevice()
        except cupy.cuda.compiler.CompileException as e:
            raise MissingDependencies(
                "CuPy fails to run due to missing CUDA Runtime.",
                pip_deps=["nvidia-cuda-runtime-cu{}".format(cuda_major)],
                conda_deps=[],
            ) from e
        except (cupy_backends.cuda.api.runtime.CUDARuntimeError, RuntimeError) as e:
            raise SoftwareException("CuPy could not execute properly.") from e

    @dataclasses.dataclass
    class NvidiaSoLib:
        """Description of a NVidia .so dynamic library"""

        soname: str
        """Dynamic library name (eg libname.so.1)"""

        module_name: str
        """Name of the nvidia module containing the library"""

        pip_pkg: typing.Optional[str] = None
        """Name of a pip-installable package providing that library"""

        conda_pkg: typing.Optional[str] = None
        """Name of a conda-installable package providing that library"""

    @staticmethod
    def _try_import_nvidia_solib(libs: typing.Sequence[NvidiaSoLib]) -> None:
        """Try to load a set of Nvidia dynamic libraries"""
        import ctypes
        import importlib
        import pathlib

        from ..utils.exceptions import ExceptionGroup

        try:
            nvidia_root = pathlib.Path(
                importlib.import_module("nvidia").__file__
            ).parent
        except ModuleNotFoundError:
            nvidia_root = None

        failed_idx: typing.List[int] = []
        exceptions = []
        for idx, lib in enumerate(libs):
            try:
                ctypes.cdll.LoadLibrary(lib.soname)
                continue
            except OSError as e:
                exceptions.append(e)

            try:
                if nvidia_root is not None:
                    ctypes.cdll.LoadLibrary(
                        nvidia_root / lib.module_name / "lib" / lib.soname
                    )
                    continue
            except OSError as e:
                exceptions.append(e)

            failed_idx.append(idx)

        if failed_idx:
            raise MissingDependencies(
                "Could not load following NVidia libraries: {}".format(
                    ", ".join([libs[idx].soname for idx in failed_idx])
                ),
                pip_deps=[
                    libs[idx].pip_pkg
                    for idx in failed_idx
                    if libs[idx].pip_pkg is not None
                ],
                conda_deps=[
                    libs[idx].conda_pkg
                    for idx in failed_idx
                    if libs[idx].conda_pkg is not None
                ],
            ) from ExceptionGroup(
                "Following exceptions were raised while trying to load NVidia libraries",
                exceptions,
            )

    @classmethod
    def check_cuda_backend(
        cls,
        name: str,
        backend_module_name: str,
        cuda_min: tuple[int, int],  # Inclusive minimum
        cuda_max: tuple[int, int],  # Exclusive maximum
        module_loader: typing.Callable[[], None],  # Method loading
        dynlib_loader: typing.Optional[typing.Callable[[], None]] = None,
    ) -> BackendMethods:
        """Perform all tests to ensure that a CUDA backend can be used"""

        # 1. Check backend module is installed
        try:
            cls._check_module_installed(name, backend_module_name)
        except BackendNotInstalled as e:
            raise MissingDependencies(
                "GBGPU CUDA plugin is missing.",
                pip_deps=["gbgpu-cuda{}x".format(cuda_min[0])],
                conda_deps=[],
            ) from e

        # 2. Check CuPy works with expected cuda major version
        cls._check_cupy_works(cuda_major=cuda_min[0])

        # 3. Try to load directly
        try:
            return module_loader()
        except BackendUnavailableException:
            pass

        # 4. Get and check the CUDA version
        cuda_version = cls._get_cuda_version()

        def fmt_version(version: tuple[int, int]) -> str:
            return "{}.{}".format(version[0], version[1])

        if cuda_version < cuda_min:
            raise MissingDriver(
                "Cuda version is below minimum supported version (expected >= {} and < {}, got {})".format(
                    fmt_version(cuda_min),
                    fmt_version(cuda_max),
                    fmt_version(cuda_version),
                )
            )
        if cuda_version >= cuda_max:
            raise MissingDriver(
                "Cuda version is above maximum supported version (expected >= {} and < {}, got {})".format(
                    fmt_version(cuda_min),
                    fmt_version(cuda_max),
                    fmt_version(cuda_version),
                )
            )

        # 5. Retry to load module
        try:
            return module_loader()
        except BackendUnavailableException as e:
            if dynlib_loader is None:
                raise e

        # 6. module_loader failed but dynlib_loader is defined, let's try that
        dynlib_loader()
        return module_loader()

    def set_cuda_device(self, dev: int):
        """Globally sets CUDA device"""
        from cupy.cuda.runtime import setDevice

        setDevice(dev)


class Cuda11xBackend(_CudaBackend, abc.ABC):
    """Implementation of CUDA 11.x backend"""

    @staticmethod
    @abc.abstractmethod
    def cuda11x_module_loader():
        raise NotImplementedError

    @staticmethod
    def cuda11x_dynlib_loader():
        import sys

        if sys.platform == "linux":
            cuda11x_solibs = [
                _CudaBackend.NvidiaSoLib(
                    soname="libcudart.so.11",
                    module_name="cuda_runtime",
                    pip_pkg="nvidia-cuda-runtime-cu11",
                    conda_pkg=None,
                ),
                _CudaBackend.NvidiaSoLib(
                    soname="libcublas.so.11",
                    module_name="cublas",
                    pip_pkg="nvidia-cublas-cu11",
                    conda_pkg=None,
                ),
                _CudaBackend.NvidiaSoLib(
                    soname="libnvJitLink.so.11",
                    module_name="nvjitlink",
                    pip_pkg="nvidia-nvjitlink-cu11",
                    conda_pkg=None,
                ),
                _CudaBackend.NvidiaSoLib(
                    soname="libcusparse.so.11",
                    module_name="cusparse",
                    pip_pkg="nvidia-cusparse-cu11",
                    conda_pkg=None,
                ),
                _CudaBackend.NvidiaSoLib(
                    soname="libnvrtc.so.11",
                    module_name="cuda_nvrtc",
                    pip_pkg="nvidia-cuda-nvrtc-cu11",
                    conda_pkg=None,
                ),
                # TODO: check this??
                # _CudaBackend.NvidiaSoLib(
                #     soname="libcufftw.so.11",
                #     module_name="cufft",
                #     pip_pkg="nvidia-cufft-cu11",
                #     conda_pkg=None,
                # ),
            ]
            _CudaBackend._try_import_nvidia_solib(cuda11x_solibs)

    def __init__(self):
        """Initialize the CPU backend"""
        
        if self.backend_name is None:
            raise ValueError("Child class must declare `backend_name` class attribute.")
        
        name = "cuda11x"
        methods = self.check_cuda_backend(
            name=name,
            backend_module_name=self.backend_name,
            cuda_min=(11, 2),
            cuda_max=(12, 0),
            module_loader=Cuda11xBackend.cuda11x_module_loader,
            dynlib_loader=Cuda11xBackend.cuda11x_dynlib_loader,
        )
        Feature = Backend.Feature

        super().__init__(
            name=name,
            methods=methods,
            features=Feature.CUPY | Feature.CUDA | Feature.GPU,
        )


class GBGPUCuda11xBackend(Cuda11xBackend, GBGPUBackend):

    """Implementation of CUDA 11.x backend"""
    backend_name : str = "gbgpu_backend_cuda11x"

    def __init__(self, *args, **kwargs):
        Cuda11xBackend.__init__(self, *args, **kwargs)
        GBGPUBackend.__init__(self, self.cuda11x_module_loader())
        
    @staticmethod
    def cuda11x_module_loader():
        try:
            import gbgpu_backend_cuda11x.utils

        except (ModuleNotFoundError, ImportError) as e:
            raise BackendUnavailableException(
                "'cuda11x' backend could not be imported."
            ) from e

        try:
            import cupy
        except (ModuleNotFoundError, ImportError) as e:
            raise MissingDependencies(
                "'cuda11x' backend requires cupy", pip_deps=["cupy-cuda11x"]
            ) from e

        return GBGPUBackendMethods(
            get_ll=gbgpu_backend_cuda11x.utils.get_ll,
            fill_global=gbgpu_backend_cuda11x.utils.fill_global,
            xp=cupy,
        )


class Cuda12xBackend(_CudaBackend, abc.ABC):
    """Implementation of CUDA 12.x backend"""

    @staticmethod
    @abc.abstractmethod
    def cuda12x_module_loader():
        raise NotImplementedError

    @staticmethod
    def cuda12x_dynlib_loader():
        import sys

        if sys.platform == "linux":
            cuda12x_solibs = [
                _CudaBackend.NvidiaSoLib(
                    soname="libcudart.so.12",
                    module_name="cuda_runtime",
                    pip_pkg="nvidia-cuda-runtime-cu12",
                    conda_pkg=None,
                ),
                _CudaBackend.NvidiaSoLib(
                    soname="libcublas.so.12",
                    module_name="cublas",
                    pip_pkg="nvidia-cublas-cu12",
                    conda_pkg=None,
                ),
                _CudaBackend.NvidiaSoLib(
                    soname="libnvJitLink.so.12",
                    module_name="nvjitlink",
                    pip_pkg="nvidia-nvjitlink-cu12",
                    conda_pkg=None,
                ),
                _CudaBackend.NvidiaSoLib(
                    soname="libcusparse.so.12",
                    module_name="cusparse",
                    pip_pkg="nvidia-cusparse-cu12",
                    conda_pkg=None,
                ),
                _CudaBackend.NvidiaSoLib(
                    soname="libnvrtc.so.12",
                    module_name="cuda_nvrtc",
                    pip_pkg="nvidia-cuda-nvrtc-cu12",
                    conda_pkg=None,
                ),
                # TODO: check this
                # _CudaBackend.NvidiaSoLib(
                #     soname="libcufftw.so.12",
                #     module_name="cufft",
                #     pip_pkg="nvidia-cufft-cu12",
                #     conda_pkg=None,
                # ),
            ]
            _CudaBackend._try_import_nvidia_solib(cuda12x_solibs)

    def __init__(self):
        """Initialize the CPU backend"""
        
        if self.backend_name is None:
            raise ValueError("Child class must declare `backend_name` class attribute.")
        
        name = "cuda12x"
        methods = self.check_cuda_backend(
            name=name,
            backend_module_name=self.backend_name,
            cuda_min=(12, 0),
            cuda_max=(13, 0),
            module_loader=self.__class__.cuda12x_module_loader,
            dynlib_loader=self.__class__.cuda12x_dynlib_loader,
        )
        Feature = Backend.Feature

        super().__init__(
            name=name,
            methods=methods,
            features=Feature.CUPY | Feature.CUDA | Feature.GPU,
        )


class GBGPUCuda12xBackend(Cuda12xBackend, GBGPUBackend):
    """Implementation of CUDA 12.x backend"""
    backend_name : str = "gbgpu_backend_cuda12x"
    
    def __init__(self, *args, **kwargs):
        Cuda12xBackend.__init__(self, *args, **kwargs)
        GBGPUBackend.__init__(self, self.cuda12x_module_loader())
        
    @staticmethod
    def cuda12x_module_loader():
        try:
            import gbgpu_backend_cuda12x.utils

        except (ModuleNotFoundError, ImportError) as e:
            raise BackendUnavailableException(
                "'cuda12x' backend could not be imported."
            ) from e

        try:
            import cupy
        except (ModuleNotFoundError, ImportError) as e:
            raise MissingDependencies(
                "'cuda12x' backend requires cupy", pip_deps=["cupy-cuda12x"]
            ) from e

        return GBGPUBackendMethods(
            get_ll=gbgpu_backend_cuda12x.utils.get_ll,
            fill_global=gbgpu_backend_cuda12x.utils.fill_global,
            xp=cupy,
        )


KNOWN_BACKENDS = {
    "cuda12x": GBGPUCuda12xBackend,
    "cuda11x": GBGPUCuda11xBackend,
    "cpu": GBGPUCpuBackend,
}

"""List of existing backends, per default order of preference."""


class BackendStatus:
    """Base class for backend statuses in the backends manager"""


class BackendStatusUnloaded(BackendStatus):
    """Backend is not yet loaded"""


@dataclasses.dataclass
class BackendStatusLoaded(BackendStatus):
    """Backend loaded successfully"""

    instance: Backend


class BackendStatusDisabled(BackendStatus):
    """Backend has been disabled by configuration options"""


@dataclasses.dataclass
class BackendStatusUnavailable:
    """Backend could not be loaded due to exception"""

    reason: BackendUnavailableException


class BackendAccessException(GBGPUException):
    """Method raised by BackendManager if requested backend cannot be accessed"""


class BackendsManager:
    """Handles loading and accessing backend instances."""

    _registry: typing.Dict[str, BackendStatus]

    @property
    def backend_list(self) -> typing.List[str]:
        """Return the list of backend names"""
        return [name for name in self._registry.keys()]

    def __init__(self, enabled_backends: typing.Optional[typing.Sequence[str]] = None):
        """
        Initialize the backend registry.

        enabled_backends: optional list of backend names which must be loaded, others will be disabled
        """
        self._registry = {
            name: BackendStatusUnloaded() for name in KNOWN_BACKENDS.keys()
        }

        if enabled_backends is not None:
            for backend_name in self.backend_list:
                if backend_name in enabled_backends:
                    status = self._try_loading_backend(backend_name)
                    if isinstance(status, BackendStatusUnavailable):
                        raise BackendAccessException(
                            "Backend '{}' is marked as enabled but cannot be loaded".format(
                                backend_name
                            )
                        ) from status.reason
                else:
                    self._disable_backend(backend_name)

    def _disable_backend(self, backend_name: str):
        """Mark a backend as disabled"""
        self._registry[backend_name] = BackendStatusDisabled()

    def _try_loading_backend(self, backend_name: str) -> BackendStatus:
        """
        Try to load a backend and return its new status.

        If current status is different from UNLOADED, return it directly.
        """
        if backend_name not in self._registry:
            raise ValueError(
                "'{}' is not a valid backend name, expected any of: {}".format(
                    backend_name, ", ".join(self.backend_list)
                )
            )

        if not isinstance(
            current_status := self._registry[backend_name], BackendStatusUnloaded
        ):
            return current_status

        try:
            new_status = BackendStatusLoaded(instance=KNOWN_BACKENDS[backend_name]())
        except BackendUnavailableException as e:
            new_status = BackendStatusUnavailable(reason=e)

        self._registry[backend_name] = new_status
        return new_status

    def get_backend(self, backend_name: str) -> Backend:
        """Get a backend instance or raise a BackendAccessException"""
        status = self._try_loading_backend(backend_name=backend_name)
        if isinstance(status, BackendStatusDisabled):
            raise BackendAccessException(
                "Backend '{}' cannot be accessed, it has been disabled.".format(
                    backend_name
                )
            )
        if isinstance(status, BackendStatusUnavailable):
            raise BackendAccessException(
                "Backend '{}' is unavailable. See previous error messages.".format(
                    backend_name
                )
            ) from status.reason

        assert isinstance(status, BackendStatusLoaded)
        return status.instance

    def has_backend(self, backend_name: str) -> bool:
        """Check if a backend is available"""
        return isinstance(
            self._try_loading_backend(backend_name=backend_name), BackendStatusLoaded
        )

    def get_first_backend(self, backends: typing.Sequence[str]) -> Backend:
        """Get first available backend from a list or raise BackendAccessException if none available"""
        assert len(backends) > 0

        reasons: typing.List[BackendUnavailableException] = []
        for backend_name in backends:
            if isinstance(
                status := self._try_loading_backend(backend_name=backend_name),
                BackendStatusLoaded,
            ):
                return status.instance
            if isinstance(status, BackendStatusUnavailable):
                reasons.append(status.reason)
            if isinstance(status, BackendStatusDisabled):
                reasons.append(
                    BackendAccessException(
                        "Backend '{}' is disabled.".format(backend_name)
                    )
                )

        from ..utils.exceptions import ExceptionGroup

        raise BackendAccessException(
            "Could not access any of the following backends which are either disabled or unavailable: {}".format(
                ", ".join(backends)
            )
        ) from ExceptionGroup(
            "The backends were not available for following reasons.", reasons
        )


__all__ = ["BackendsManager", "BackendAccessException"]
