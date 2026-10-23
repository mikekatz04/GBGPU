"""JAX implementations of GB-specific waveform / WDM / heterodyne machinery.

Absorbed from ``fastlisaresponse.jax`` at Phase 3F of the sprint reorg.
Subpackages:

* :mod:`gbgpu.jax.sources` — JAX source classes (:class:`~gbgpu.jax.sources.ucb.JaxUCBSource`).
* :mod:`gbgpu.jax.wdm`     — WDM-domain JAX kernels (``gb_wdm_*``, ``gb_wdm_het_*``).

Modules imported on demand (not by this package): :mod:`gbgpu.jax.tdi_on_the_fly` (the
functional TDI-on-the-fly kernels, e.g. ``gb_run_wave_tdi``), :mod:`gbgpu.jax.wrappers`
and :mod:`gbgpu.jax.wdm.computation_group` / :mod:`gbgpu.jax.wdm.signal_het_kernels`.

Generic LISA-physics JAX (response, orbits, WDM lookup tables) lives in
:mod:`lisatools.jax` (LISAanalysistools). Subpackage import is gated on
``import jax``: without JAX, ``gbgpu.jax`` imports with an empty ``__all__``. GBGPU
registers no ``gbgpu_jax`` backend, and the 9mo / 12mo global-fit runs do not import this
package (their GB likelihoods run on the C++ / CUDA backends).
"""
from __future__ import annotations

try:
    import jax  # noqa: F401
    import jax.numpy as jnp  # noqa: F401
    _HAS_JAX = True
except (ImportError, ModuleNotFoundError):
    _HAS_JAX = False

if _HAS_JAX:
    from . import sources, wdm

    __all__ = ["sources", "wdm"]
else:
    __all__ = []
