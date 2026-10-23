"""GB source-class JAX implementations.

Exports :class:`~gbgpu.jax.sources.ucb.JaxUCBSource`, the JAX port of the C++
``GBTDIonTheFly`` UCB amplitude / phase, for the LAT JAX response
(:mod:`lisatools.jax.response`).
"""
from __future__ import annotations

from .ucb import JaxUCBSource

__all__ = ["JaxUCBSource"]
