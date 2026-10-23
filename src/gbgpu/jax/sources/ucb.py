"""JAX UCB (galactic-binary) source.

1:1 JAX port of ``GBTDIonTheFly::ucb_{amplitude,phase,f,fdot}`` (GBGPU
``cutils/gb_tdi_on_the_fly.cu``). Parameter layout matches the C++ class::

    params[0] = amplitude  (A0)
    params[1] = f0         (Hz, at t_ref)
    params[2] = fdot       (Hz/s)
    params[3] = fddot      (Hz/s^2)
    params[4] = phi0       (rad)
    params[5] = inc        (rad)
    params[6] = psi        (rad)
    params[7] = lam        (rad, sky longitude)
    params[8] = beta       (rad, sky latitude)

The sky angles are read by the LAT JAX projection
(:func:`lisatools.jax.response.projection.get_sky_vectors`) against the orbit positions,
so they are in the frame of the orbits used (ecliptic for ecliptic-frame orbits; ICRS
right ascension / declination for ICRS orbits, as on the global fit).

Same formulas as ``GBTDIonTheFly::ucb_*``. ``t_ref`` is fixed at construction.
"""
from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp

from lisatools.jax.response.base import JaxAmpPhaseSource


class JaxUCBSource(JaxAmpPhaseSource):
    """Galactic-binary monochromatic-with-chirp source.

    A static (configuration-only) :class:`lisatools.jax.response.base.JaxAmpPhaseSource`:
    per-source parameters are passed to each method as a flat length-9 array
    (``n_params``), in the ``param_names`` order ``(amp, f0, fdot, fddot, phi0, inc, psi,
    lam, beta)``, so ``jax.vmap`` over the binary axis drives a batch. Used by the GBGPU
    JAX kernels (:mod:`gbgpu.jax.tdi_on_the_fly`, :mod:`gbgpu.jax.wdm`).

    Args:
        t_ref: GB phase reference time (seconds). Same as
            ``GBTDIonTheFly::t_ref`` in C++.

    Attributes:
        inc_index, psi_index, lam_index, beta_index (int): 5, 6, 7, 8 (read by the
            projection through ``get_sky_indices``).
        t_ref (float): The reference time [s].

    Example:
        >>> import jax.numpy as jnp
        >>> from gbgpu.jax.sources.ucb import JaxUCBSource
        >>> src = JaxUCBSource(t_ref=0.0)
        >>> p = jnp.array([1.0, 0.25, 0.0, 0.0, 0.5, 1.0, 0.3, 2.0, 0.4])
        >>> float(src.f(0.0, p)), float(src.amplitude(0.0, p))
        (0.25, 1.0)
    """

    n_params: int = 9
    param_names: Tuple[str, ...] = (
        "amp", "f0", "fdot", "fddot", "phi0",
        "inc", "psi", "lam", "beta",
    )
    inc_index: int = 5
    psi_index: int = 6
    lam_index: int = 7
    beta_index: int = 8

    # C++ indices
    amplitude_index = 0
    f0_index = 1
    fdot0_index = 2
    fddot0_index = 3
    phi0_index = 4

    def __init__(self, t_ref: float):
        self.t_ref = float(t_ref)

    def amplitude(self, t: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        r"""Strain amplitude with the chirp correction.

        .. math::

            A(t) = A_0 \left(1 + \frac{2}{3}\frac{\dot f}{f_0}(t - t_{ref})\right)

        Args:
            t: Time(s) [s], scalar or array.
            params: One source's length-9 parameter array.

        Returns:
            jnp.ndarray: Amplitude, broadcast with ``t``.
        """
        # Mirrors GBTDIonTheFly::ucb_amplitude (cutils/gb_tdi_on_the_fly.cu).
        A0 = params[self.amplitude_index]
        f0 = params[self.f0_index]
        fdot = params[self.fdot0_index]
        t_diff = t - self.t_ref
        return A0 * (1.0 + (2.0 / 3.0) * (fdot / f0) * t_diff)

    def phase(self, t: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        r"""GW phase :math:`-\phi_0 + 2\pi(f_0 \tau + \dot f \tau^2 / 2 + \ddot f \tau^3 / 6)`.

        ``tau = t - t_ref``; the leading ``-phi0`` is the LDC convention of the C++ class.

        Args:
            t: Time(s) [s], scalar or array.
            params: One source's length-9 parameter array.

        Returns:
            jnp.ndarray: Phase [rad], broadcast with ``t``.
        """
        # Mirrors GBTDIonTheFly::ucb_phase (cutils/gb_tdi_on_the_fly.cu). Note the LDC ``-phi0``.
        f0 = params[self.f0_index]
        phi0 = params[self.phi0_index]
        fdot = params[self.fdot0_index]
        fddot = params[self.fddot0_index]
        t_diff = t - self.t_ref
        return (
            -phi0
            + 2.0 * jnp.pi * (
                f0 * t_diff
                + 0.5 * fdot * t_diff * t_diff
                + (1.0 / 6.0) * fddot * t_diff * t_diff * t_diff
            )
        )

    # ---- convenience helpers (not required by projection) ----
    def f(self, t: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        """Instantaneous GW frequency ``f0 + fdot tau + fddot tau^2 / 2`` [Hz], ``tau = t - t_ref``.

        Args:
            t: Time(s) [s].
            params: One source's length-9 parameter array.

        Returns:
            jnp.ndarray: Frequency [Hz], broadcast with ``t``.
        """
        f0 = params[self.f0_index]
        fdot = params[self.fdot0_index]
        fddot = params[self.fddot0_index]
        t_diff = t - self.t_ref
        return f0 + fdot * t_diff + 0.5 * fddot * t_diff * t_diff

    def fdot(self, t: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        """Frequency derivative ``fdot + fddot (t - t_ref)`` [Hz/s].

        Args:
            t: Time(s) [s].
            params: One source's length-9 parameter array.

        Returns:
            jnp.ndarray: Frequency derivative [Hz/s], broadcast with ``t``.
        """
        fdot = params[self.fdot0_index]
        fddot = params[self.fddot0_index]
        t_diff = t - self.t_ref
        return fdot + fddot * t_diff
