"""GBGPU: galactic-binary waveforms and likelihoods for LISA on CPUs and GPUs.

The GB-physics package of LISA Analysis Tools, built on ``lisatools`` (orbits, TDI, domains,
the chunked-heterodyne base) and ``gpubackendtools`` (backend machinery). Main entry points:

* :class:`gbgpu.gbgpu.GBGPU`: FastGB frequency-domain waveforms
  (:meth:`~gbgpu.gbgpu.GBGPUBase.run_wave`) and the legacy FD likelihood / global-template
  kernels;
  :class:`gbgpu.thirdbody.GBGPUThirdBody` adds a third body.
* :class:`gbgpu.gbcomps.GBWDMComputations`: the WDM-domain chunked-heterodyne GB likelihood
  of the global fit; :class:`gbgpu.gbcomps.GBFDComputations` its FD analog.
* :class:`gbgpu.gbsignalhetcomputations.GBSignalHetComputations`: the signal-heterodyne
  in-model scorer wrapped around the chunked comp (``for_band_engine``).
* :class:`gbgpu.gblookupcomputations.GBLookupComputations`: the direct-to-WDM lookup scorer
  (reference, anchor and information matrix of the sig-het engine).

Importing the package registers the backends ``gbgpu_cpu``, ``gbgpu_cuda11x``,
``gbgpu_cuda12x`` and ``gbgpu_cuda13x`` (:mod:`gbgpu.cutils`) with the gpubackendtools
registry. :func:`get_backend`, :func:`has_backend` and :func:`get_first_backend` are
``gbgpu_``-prefixing facades over the gpubackendtools functions; :func:`get_include` and
:func:`get_cmake_module_path` locate the C++/CUDA headers for downstream builds.

Example:
    .. code-block:: python

        import gbgpu

        gbgpu.get_backend("cpu").name  # "gbgpu_cpu"
        gbgpu.has_backend("gbgpu_cuda13x")
"""

# ruff: noqa: E402
try:
    from gbgpu._version import (  # pylint: disable=E0401,E0611
        __version__,
        __version_tuple__,
    )

except ModuleNotFoundError:
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover

    try:
        __version__ = version(__name__)
        __version_tuple__ = tuple(__version__.split("."))
    except PackageNotFoundError:  # pragma: no cover
        __version__ = "unknown"
        __version_tuple__ = (0, 0, 0, "unknown")
    finally:
        del version, PackageNotFoundError

_is_editable: bool
try:
    from . import _editable

    _is_editable = True
    del _editable
except (ModuleNotFoundError, ImportError):
    _is_editable = False


def get_include() -> str:
    """Absolute path to GBGPU's public C++/CUDA header directory.

    Downstream sprint packages and lisa-on-gpu (during the Phase 3L.7
    transition) add this to their compiler include path so that::

        #include "gb_tdi_on_the_fly.hh"    // GBTDIonTheFly + GBComputationGroup

    resolves against the installed wheel.

    Pair with ``gpubackendtools.get_include()`` and
    ``lisatools.get_include()`` -- a downstream typically needs all three:
    GBT for ``gbt_global.h`` + ``cuda_complex.hpp`` + ``InterpolateDevice.hh``,
    LAT for ``Detector.hpp`` + ``lat_chunked_het_kernels.hh`` + base
    classes, GBGPU for the GB source-class-specific machinery
    (``GBTDIonTheFly``, ``GBComputationGroup``, ``gb_run_wave_tdi_wrap``).

    Returns:
        str: The ``cutils`` directory of the installed ``gbgpu`` package.

    Example:
        .. code-block:: bash

            nvcc -I"$(python -c 'import gbgpu; print(gbgpu.get_include())')" ...
    """
    import os.path
    return os.path.join(os.path.dirname(__file__), "cutils")


def get_cmake_module_path() -> str:
    """Absolute path to the directory containing ``GBGPUConfig.cmake``.

    For downstreams that prefer ``find_package(GBGPU CONFIG)`` over the
    Python shell-out form of :func:`get_include`. Forward-compatible:
    ``GBGPUConfig.cmake`` ships at the cutils directory once added;
    consumers can begin using this path before that file exists (CMake
    just won't find the package and they fall through to the shell-out
    form).

    Returns:
        str: The same directory as :func:`get_include`.
    """
    import os.path
    return os.path.join(os.path.dirname(__file__), "cutils")


from . import cutils, utils

from gpubackendtools import Globals
from .cutils import (
    GBGPUCpuBackend,
    GBGPUCuda11xBackend,
    GBGPUCuda12xBackend,
    GBGPUCuda13xBackend,
)


# Phase 3L.7k (2026-06-04): GBGPU's backends now compose LAT's
# LISAToolsBackend surface with GB-specific Wraps. The previous
# fastlisaresponse_<flavor> backend family (lisa-on-gpu) is being
# retired and consumers can use these directly via
# ``gbgpu.get_backend(...)``.
add_backends = {
    "gbgpu_cpu": GBGPUCpuBackend,
    "gbgpu_cuda11x": GBGPUCuda11xBackend,
    "gbgpu_cuda12x": GBGPUCuda12xBackend,
    "gbgpu_cuda13x": GBGPUCuda13xBackend,
}

Globals().backends_manager.add_backends(add_backends)


from gpubackendtools import get_backend as _get_backend
from gpubackendtools import has_backend as _has_backend
from gpubackendtools import get_first_backend as _get_first_backend
from gpubackendtools.gpubackendtools import Backend


def get_backend(backend: str) -> Backend:
    """Get a GBGPU backend by flavor or full name.

    A name without ``"gbgpu_"`` in it is prefixed (``"cpu"`` -> ``"gbgpu_cpu"``), then
    passed to :func:`gpubackendtools.globals.get_backend`, which also resolves the
    ``"gbgpu_cuda"`` / ``"gbgpu_gpu"`` aliases to the first available CUDA flavor. (The
    ``__doc__ = ...`` line below only sets a local variable.)

    Args:
        backend: ``"cpu"``, ``"cuda11x"``, ``"cuda12x"``, ``"cuda13x"``, ``"cuda"``,
            ``"gpu"``, or a full ``"gbgpu_*"`` name.

    Returns:
        Backend: The loaded backend singleton (``xp``, the LAT and GB native wraps).

    Raises:
        ValueError: If the resolved name is not registered.
        BackendAccessException: If the backend is disabled or failed to load.

    Example:
        .. code-block:: python

            import gbgpu

            backend = gbgpu.get_backend("cpu")
            backend.xp  # numpy
    """
    __doc__ = _get_backend.__doc__
    if "gbgpu_" not in backend:
        return _get_backend("gbgpu_" + backend)
    else:
        return _get_backend(backend)


def has_backend(backend: str) -> Backend:
    """Test whether a GBGPU backend loads.

    A name without ``"gbgpu_"`` is prefixed, then passed to
    :func:`gpubackendtools.globals.has_backend` (loads it on first call).

    Args:
        backend: A flavor (``"cpu"``, ``"cuda13x"``, ...) or a full ``"gbgpu_*"`` name. The
            ``"cuda"`` / ``"gpu"`` aliases are NOT resolved here (``"gbgpu_cuda"`` is not a
            registry name and raises ``ValueError``).

    Returns:
        bool: ``True`` if the backend loaded (despite the ``Backend`` annotation).

    Raises:
        ValueError: If the resolved name is not a registry name.
    """
    __doc__ = _has_backend.__doc__
    if "gbgpu_" not in backend:
        return _has_backend("gbgpu_" + backend)
    else:
        return _has_backend(backend)

        
def get_first_backend(backend: str) -> Backend:
    """Intended: the first available GBGPU backend of a candidate list (does not work).

    Prefixes ``"gbgpu_"`` and calls :func:`gpubackendtools.globals.get_first_backend`,
    which expects a SEQUENCE of names. As written a string argument becomes one prefixed
    string that is iterated character by character (``"g"`` is not a backend name), and a
    list argument fails at the ``"gbgpu_" + backend`` concatenation (``TypeError``). Use
    ``gpubackendtools.get_first_backend(["gbgpu_cuda13x", "gbgpu_cpu"])`` instead.

    Args:
        backend: Candidate name(s).

    Returns:
        Backend: Not reached for valid inputs (see above).
    """
    __doc__ = _get_first_backend.__doc__
    if "gbgpu_" not in backend:
        return _get_first_backend("gbgpu_" + backend)
    else:
        return _get_first_backend(backend)


__all__ = [
    "__version__",
    "__version_tuple__",
    "_is_editable",
    # "amplitude",
    # "cutils",
    # "files",
    # "summation",
    # "trajectory",
    # "utils",
    # "waveform",
    "get_logger",
    "get_config",
    "get_config_setter",
    "get_backend",
    "get_file_manager",
    "has_backend",
]
