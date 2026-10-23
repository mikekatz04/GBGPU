"""GBGPU dispatch base.

Phase 3L.7k (2026-06-04): GBGPU-specific subclass of
``ParallelModuleBase`` that prefixes ``gbgpu_`` onto bare backend tags
(``cpu`` / ``cuda13x`` / ...) so GBGPU compute objects resolve to the
GBGPU-composed backend (LAT response Wraps + GB-specific Wraps in one
object, :mod:`gbgpu.cutils`). Parallels
:class:`lisatools.utils.parallelbase.LISAToolsParallelModule`, which prefixes
``lisatools_``.

Subclasses in GBGPU: :class:`gbgpu.gbgpu.GBGPUBase`, :class:`gbgpu.gbcomps.GBFDComputations`,
:class:`gbgpu.gblookupcomputations.GBLookupComputations` and
:class:`gbgpu.gbsignalhetcomputations.GBSignalHetComputations`.
:class:`gbgpu.gbcomps.GBWDMComputations` derives from the LAT base instead and sets
``_BACKEND_PREFIX = "gbgpu"`` there. ``gbgpu.utils.parallelbase`` re-exports this class.
"""

from gpubackendtools import ParallelModuleBase


class GBGPUParallelModule(ParallelModuleBase):
    """Base class for GBGPU parallel-dispatch modules.

    Subclasses (``GBGPUBase`` / ``GBFDComputations`` / etc.)
    inherit from this and pass ``force_backend="cpu"`` style strings.
    The string is automatically prefixed with ``gbgpu_`` so the
    resolved backend is ``gbgpu_cpu`` (which carries both LAT and GB
    native symbols after Phase 3L.7k). The backend is chosen once, at construction;
    methods dispatch through ``self.backend`` / ``self.xp``.

    Args:
        force_backend (str | Backend | tuple, optional): A string without the ``gbgpu_``
            prefix (``"cpu"``, ``"cuda11x"``, ``"cuda12x"``, ``"cuda13x"``, or the
            ``"cuda"`` / ``"gpu"`` aliases for the first available CUDA flavor) becomes
            ``("gbgpu", force_backend)``; a ``"gbgpu_*"`` string, a backend object or a
            ``(package, flavor)`` tuple is passed to
            :class:`gpubackendtools.parallelbase.ParallelModuleBase` unchanged. Default
            ``None`` takes the first loadable entry of the subclass's
            ``supported_backends()`` (which the subclass must define).

    Raises:
        ValueError: If the resolved name is not a registered backend.
        BackendAccessException: If the backend is disabled or unavailable.

    Example:
        .. code-block:: python

            from gbgpu.parallelbase import GBGPUParallelModule


            class MyGBComp(GBGPUParallelModule):
                @classmethod
                def supported_backends(cls):
                    return ["gbgpu_" + flavor for flavor in cls.GPU_RECOMMENDED()]


            comp = MyGBComp(force_backend="cpu")
            comp.backend.name  # "gbgpu_cpu"
    """

    _BACKEND_PREFIX = "gbgpu"

    def __init__(self, force_backend=None):
        if isinstance(force_backend, str) and not force_backend.startswith(
            self._BACKEND_PREFIX + "_"
        ):
            force_backend = (self._BACKEND_PREFIX, force_backend)
        super().__init__(force_backend)

    # TODO check: keep/remove/document -- no callers; no gbgpu_jax backend is registered
    @staticmethod
    def GPU_RECOMMENDED_WITH_JAX() -> list[str]:
        """Same as GPU_RECOMMENDED() but with the JAX backend appended (bare flavors)."""
        return ["cuda13x", "cuda12x", "cuda11x", "cpu", "jax"]
