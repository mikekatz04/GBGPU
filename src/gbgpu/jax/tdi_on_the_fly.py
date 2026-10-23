"""Functional JAX kernels mirroring ``gb_run_wave_tdi_wrap`` and
``sobbh_run_wave_tdi_wrap`` from the C++ side.

These are *pure* functions of (params, t_arr, source, orbits,
tdi_config) -- they return new arrays rather than mutating buffers, so
``jax.grad`` / ``jax.jit`` / ``jax.vmap`` work without surprises. The
``GBTDIonTheFlyWrapJAX`` / ``SOBBHTDIonTheFlyWrapJAX`` thin objects in
``wrappers.py`` adapt this to the C++ wrapper API.

Entry points: :func:`run_wave_tdi` (any :class:`JaxAmpPhaseSource`) and
:func:`gb_run_wave_tdi` (the UCB source; used in the LATW tutorial
``08_BackendsAndDevWorkflow``). ``orbits`` / ``tdi_config`` are the LAT JAX wraps
:class:`lisatools.jax.orbits.OrbitsWrapJAX` /
:class:`lisatools.jax.response.tdi_config.TDIConfigWrapJAX`; the sky angles are read in
the orbits' frame.
"""
from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

# Phase 3L.7j (2026-06-04): JAX TDI-on-the-fly carved out of
# lisa-on-gpu's fastlisaresponse.jax.tdi_on_the_fly. Generic-LISA-response
# helpers (amp_phase_extract, base, orbits, projection, tdi_config) live
# in LAT post-Phase-3D; this file is GB+SOBBH-flavored entry points on
# top of them. SOBBH variant is included alongside GB because the only
# differences are the source class swap-in and a per-source param
# layout (mirrors the gbcomps.py pattern).
from lisatools.jax.orbits import OrbitsWrapJAX
from lisatools.jax.response.amp_phase_extract import extract_and_unwrap
from lisatools.jax.response.base import JaxAmpPhaseSource
from lisatools.jax.response.projection import get_phase_ref, get_sky_vectors, get_tdi_Xf_single
from lisatools.jax.response.tdi_config import TDIConfigWrapJAX


def _run_single_binary(params: jnp.ndarray, t_arr: jnp.ndarray,
                       source: JaxAmpPhaseSource,
                       orbits: OrbitsWrapJAX,
                       tdi_config: TDIConfigWrapJAX
                       ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate the (M, tdi_amp, tdi_phase, phase_ref) block for one source.

    Returns:
        tuple: ``(M, tdi_amp, tdi_phase, phase_ref)``: ``M`` complex ``(nchannels, N)``,
        the raw complex TDI samples; ``tdi_amp`` real ``(nchannels, N)``, signed
        amplitudes; ``tdi_phase`` real ``(nchannels, N)``, unwrapped phases;
        ``phase_ref`` real ``(N,)``, the SC1 reference phase.
    """
    # Sky vectors (per-source, constant in t).
    k, u, v = get_sky_vectors(params, source)

    # Project at every time. The projection returns shape (N, nchannels).
    proj = get_tdi_Xf_single(t_arr, params, source, orbits, tdi_config, k, u, v)
    # Re-shape to (nchannels, N) to match the C++ memory layout used by
    # the WDM / spline downstream kernels.
    M = jnp.transpose(proj, (1, 0))

    # Reference phase at spacecraft 1.
    phase_ref = get_phase_ref(t_arr, params, source, orbits)

    # Extract per-channel (amp, unwrapped phase).
    tdi_amp, tdi_phase = extract_and_unwrap(M, phase_ref)
    return M, tdi_amp, tdi_phase, phase_ref


def run_wave_tdi(params: jnp.ndarray, t_arr: jnp.ndarray,
                 source: JaxAmpPhaseSource,
                 orbits: OrbitsWrapJAX,
                 tdi_config: TDIConfigWrapJAX
                 ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Functional analog of ``LISATDIonTheFly::run_wave_tdi``.

    Per binary (``jax.vmap``): the complex TDI samples ``M`` of the LAT JAX projection,
    the spacecraft-1 reference phase, and the per-channel amplitude / unwrapped phase
    extracted from ``M`` (:func:`lisatools.jax.response.amp_phase_extract.extract_and_unwrap`).

    Args:
        params: Shape ``(num_bin, n_params)`` OR flat ``(num_bin * n_params,)``, in
            ``source``'s layout.
        t_arr: Times [s], shape ``(num_bin, N)``, ``(1, N)`` or ``(N,)`` (broadcast to
            every binary).
        source: The amp / phase source (``source.n_params`` columns).
        orbits: JAX orbits.
        tdi_config: JAX TDI configuration.

    Returns:
        tuple: ``(M, tdi_amp, tdi_phase, phase_ref)`` with shapes ``M`` complex
        ``(num_bin, nchannels, N)``, ``tdi_amp`` / ``tdi_phase`` real
        ``(num_bin, nchannels, N)`` (signed amplitudes, unwrapped phases) and
        ``phase_ref`` real ``(num_bin, N)``.

    Raises:
        ValueError: ``params`` or ``t_arr`` is not 1-D or 2-D.
        AssertionError: Shapes inconsistent with ``source.n_params`` / ``num_bin``.
    """
    n_params = source.n_params
    params = jnp.asarray(params)
    if params.ndim == 1:
        assert params.shape[0] % n_params == 0
        params = params.reshape(-1, n_params)
    elif params.ndim == 2:
        assert params.shape[1] == n_params
    else:
        raise ValueError(f"params must be 1D or 2D, got shape {params.shape}.")
    num_bin = params.shape[0]

    t_arr = jnp.asarray(t_arr)
    if t_arr.ndim == 1:
        t_arr = jnp.broadcast_to(t_arr, (num_bin, t_arr.shape[0]))
    elif t_arr.ndim == 2:
        if t_arr.shape[0] == 1:
            t_arr = jnp.broadcast_to(t_arr, (num_bin, t_arr.shape[1]))
    else:
        raise ValueError(f"t_arr must be 1D or 2D, got shape {t_arr.shape}.")
    assert t_arr.shape[0] == num_bin

    # ``vmap`` over the binary axis. The projection itself is
    # constructed at trace time (TDIConfig unrolled), so vmap turns the
    # whole pipeline into one fused computation.
    def per_bin(p, t):
        return _run_single_binary(p, t, source, orbits, tdi_config)

    Ms, amps, phases, prefs = jax.vmap(per_bin)(params, t_arr)
    return Ms, amps, phases, prefs


# --------------------------------------------------------------------
# Public source-specific convenience kernels matching C++ names.
# --------------------------------------------------------------------
def gb_run_wave_tdi(params: jnp.ndarray, t_arr: jnp.ndarray,
                    t_ref: float,
                    orbits: OrbitsWrapJAX,
                    tdi_config: TDIConfigWrapJAX
                    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JAX analog of ``gb_run_wave_tdi_wrap``: :func:`run_wave_tdi` with the UCB source.

    Builds a :class:`~gbgpu.jax.sources.ucb.JaxUCBSource` at ``t_ref`` on every call.

    Args:
        params: ``(num_bin, 9)`` or ``(9 * num_bin,)`` UCB parameters per binary; layout
            from :class:`~gbgpu.jax.sources.ucb.JaxUCBSource`.
        t_arr: Times [s], ``(num_bin, N)`` or ``(N,)``.
        t_ref: UCB reference time [s].
        orbits: JAX orbits (:class:`lisatools.jax.orbits.OrbitsWrapJAX`).
        tdi_config: JAX TDI configuration
            (:class:`lisatools.jax.response.tdi_config.TDIConfigWrapJAX`).

    Returns:
        tuple: ``(M, tdi_amp, tdi_phase, phase_ref)`` -- same shape contract as
        :func:`run_wave_tdi`.

    Example:
        .. code-block:: python

            import jax
            from gbgpu.jax.tdi_on_the_fly import gb_run_wave_tdi

            jax.config.update("jax_enable_x64", True)
            # orbits_jax / tdi_jax: the JAX orbit and TDI wraps on the jax backend
            M, amp, phase, phase_ref = gb_run_wave_tdi(theta[None, :], t_arr[None, :],
                                                       t_ref, orbits_jax, tdi_jax)
            h_td = M[0].real                     # (nchannels, N)
    """
    from .sources.ucb import JaxUCBSource
    source = JaxUCBSource(t_ref=t_ref)
    return run_wave_tdi(params, t_arr, source, orbits, tdi_config)


# TODO check: keep/remove/document -- broken: imports gbgpu.jax.sources.sobbh, which does not exist
def sobbh_run_wave_tdi(params: jnp.ndarray, t_arr: jnp.ndarray,
                       t_ref: float,
                       orbits: OrbitsWrapJAX,
                       tdi_config: TDIConfigWrapJAX
                       ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JAX analog of ``sobbh_run_wave_tdi_wrap``.

    See :func:`gb_run_wave_tdi`; the only difference is the source
    plugin and parameter count (11 instead of 9).
    """
    from .sources.sobbh import JaxSOBBHSource
    source = JaxSOBBHSource(t_ref=t_ref)
    return run_wave_tdi(params, t_arr, source, orbits, tdi_config)
