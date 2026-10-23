"""GBGPU utilities: waveform helpers, citations, exceptions and the dispatch-base shim.

Submodules (none is imported by this package ``__init__``):

* :mod:`gbgpu.utils.utility` -- GB astrophysics helpers (:func:`~gbgpu.utils.utility.get_N`,
  :func:`~gbgpu.utils.utility.get_fdot`, :func:`~gbgpu.utils.utility.get_amplitude`, ...)
  and the ``lisatools.utils.constants`` re-export.
* :mod:`gbgpu.utils.citation` -- BibTeX strings used by the ``citation`` properties.
* :mod:`gbgpu.utils.exceptions` -- the GBGPU exception classes (star-imported into
  :mod:`gbgpu.cutils`).
* :mod:`gbgpu.utils.parallelbase` -- re-export of
  :class:`gbgpu.parallelbase.GBGPUParallelModule`.
* :mod:`gbgpu.utils.citations`, :mod:`gbgpu.utils.config` -- template leftovers that
  nothing imports (``config`` cannot be imported).
"""
