"""JAX ports of the direct (wavelet-lookup) ``gb_wdm_{get_ll,fill_global,swap_ll}`` kernels.

Each kernel takes a binary's parameters and the WDM domain / lookup
table data, and produces either accumulators ``(d|h), (h|h)`` or a
scattered template buffer. All kernels are pure functional ->
``jax.vmap`` over the binary axis gives batched evaluation.

The templates are built pixel by pixel from the wavelet lookup table
(:class:`lisatools.jax.wdm.wavelet_lookup.WaveletLookupTableWrapJAX`) at the
instantaneous frequency of each channel, as the former C++ direct-WDM kernels did.
Those C++ kernels are no longer bound in GBGPU (the C++ WDM path is the
chunked-heterodyne ``gb_wdm_het_*`` family, mirrored in
:mod:`gbgpu.jax.wdm.heterodyne_kernels`); the only in-package caller of this module
is :class:`gbgpu.jax.wdm.computation_group.GBComputationGroupWrapJAX`.

Inner-product normalisation matches the C++ exactly:

.. code-block:: text

    (a|b) = sum over (c_i, c_j, m, n)  of  a[m,n,c_i] * b[m,n,c_j] * invC[m,n,c_i,c_j]

(The C++ kernel internally multiplies per-pixel by 0.25 and the final
block reduction by 4.0 -- net unit prefactor. We collapse that to a
single :func:`jnp.sum` here.)

``num_diff`` (default ``2``, matching the C++ template instantiation
``gb_wdm_*_kernel<2, 5>``): per (n, c) the kernel scatters
contributions to ``num_diff`` neighbouring ``m`` layers on each side
of ``layer_m = int(f / layer_df)``, i.e. ``total_diff = 2*num_diff + 1``
layers per pixel. Setting ``num_diff = 0`` recovers the
single-layer-per-pixel mode used in the early prototypes.
"""
from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from lisatools.jax.response.base import JaxAmpPhaseSource
from lisatools.jax.orbits import OrbitsWrapJAX
from lisatools.jax.response.projection import get_sky_vectors
from lisatools.jax.response.tdi_config import TDIConfigWrapJAX
from lisatools.jax.wdm.fast_inner import fast_wdm_inner_jax
from lisatools.jax.wdm.wavelet_lookup import WaveletLookupTableWrapJAX
from lisatools.jax.wdm.wdm_domain import WDMDomainWrapJAX


#: C++ tdi_type enum values (``TDI_XYZ`` / ``TDI_AET`` / ``TDI_AE`` of the C++
#: TDI-on-the-fly headers; ``backend.TDITypeDict``).
TDI_XYZ = 1
TDI_AET = 2
TDI_AE = 3

#: Default num_diff: matches the former C++ ``gb_wdm_*_kernel<2, 5>``
#: instantiation (5 layers per pixel).
DEFAULT_NUM_DIFF = 2


def _w_mn_for_one_binary(params: jnp.ndarray,
                         source: JaxAmpPhaseSource,
                         orbits: OrbitsWrapJAX,
                         tdi_config: TDIConfigWrapJAX,
                         wdm_lookup: WaveletLookupTableWrapJAX,
                         T: float, t_ref: float,
                         deriv_delta_t: float,
                         num_diff: int = DEFAULT_NUM_DIFF) -> jnp.ndarray:
    """Build the per-pixel WDM coefficient block ``w[c, mf, n]``.

    Shape: ``(nchannels, Nf_active, Nt_active)``. Matches the C++
    template ``[c * Nf_active * Nt_active + (m - ind_min_f) *
    Nt_active + (n - ind_min_t)]``.

    For each (n, c) pixel, contributions are scattered into the
    ``2*num_diff+1`` layers ``[layer_m - num_diff, ..., layer_m +
    num_diff]`` -- with ``layer_m = int(f[c] / layer_df)``. Pixels
    that fall outside the active band ``[ind_min_f, ind_max_f)`` are
    silently dropped (matches the C++ guard).

    Pixel ``n`` (active time index offset by ``ind_min_t``) sits at
    ``t_n = n * layer_dt + t_ref``; the per-pixel response comes from
    :func:`lisatools.jax.wdm.fast_inner.fast_wdm_inner_jax`. ``T`` is accepted for
    signature parity and not used.
    """
    nch = wdm_lookup.num_channel
    Nf_act = wdm_lookup.Nf_active
    Nt_act = wdm_lookup.Nt_active
    layer_df = wdm_lookup.layer_df
    layer_dt = wdm_lookup.layer_dt
    ind_min_t = wdm_lookup.ind_min_t
    ind_min_f = wdm_lookup.ind_min_f
    ind_max_f = wdm_lookup.ind_max_f

    # Sky vectors per source (constant in t).
    k, u, v = get_sky_vectors(params, source)

    n_indices = jnp.arange(ind_min_t, ind_min_t + Nt_act)
    tn_arr = n_indices.astype(jnp.float64) * layer_dt + t_ref

    # Build (M', f, fdot) for every n via vmap. M_prime shape: (Nt_act, nch).
    def per_n(tn):
        return fast_wdm_inner_jax(tn, params, source, orbits, tdi_config,
                                  k, u, v, deriv_delta_t=deriv_delta_t)
    M_prime, f, fdot, _ = jax.vmap(per_n)(tn_arr)

    # Per-(n, c) anchor layer.
    layer_m = jnp.floor(f / layer_df).astype(jnp.int32)   # (Nt_act, nch)

    template = jnp.zeros((nch, Nf_act, Nt_act), dtype=jnp.float64)
    n_grid = jnp.arange(Nt_act)
    n_codes = n_grid + ind_min_t

    # Per channel, sweep the [-num_diff, +num_diff] window of m offsets.
    for c in range(nch):
        m_c = layer_m[:, c]

        for diff in range(-num_diff, num_diff + 1):
            m_here = m_c + diff
            mf_idx = m_here - ind_min_f
            in_band = (m_here >= ind_min_f) & (m_here < ind_max_f)
            mf_safe = jnp.clip(mf_idx, 0, Nf_act - 1)

            def per_pixel(M_c_v, f_c_v, fdot_c_v, m_v, n_v):
                return wdm_lookup.get_w_mn_lookup(M_c_v, f_c_v, fdot_c_v,
                                                  m_v, n_v)
            w_vals = jax.vmap(per_pixel)(
                M_prime[:, c], f[:, c], fdot[:, c], m_here, n_codes
            )
            w_vals = jnp.where(in_band, w_vals, 0.0)

            # Scatter-add into template[c, mf_safe, n_grid]. Pixels
            # masked-out have w_vals == 0 so they add nothing.
            template = template.at[c, mf_safe, n_grid].add(w_vals)

    return template


def _inner_product_xyz(d: jnp.ndarray, w: jnp.ndarray, invC: jnp.ndarray
                       ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """XYZ cross-channel inner product.

    Inputs:
        d   : (nch, Nf_act, Nt_act)         observed data
        w   : (nch, Nf_act, Nt_act)         template
        invC: (nch, nch, Nf_act, Nt_act)    cross-channel inverse noise
    Returns ``(d_h, h_h)`` scalars.
    """
    # Net prefactor is 1.0 (C++ uses 0.25 per pixel then 4.0 final).
    d_h = jnp.einsum('i...,ij...,j...->', d, invC, w)
    h_h = jnp.einsum('i...,ij...,j...->', w, invC, w)
    return d_h, h_h


def _inner_product_diag(d: jnp.ndarray, w: jnp.ndarray, invN: jnp.ndarray,
                        nch: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """AET / AE diagonal inner product (channels ``[:nch]``)."""
    d_sl = d[:nch]
    w_sl = w[:nch]
    N_sl = invN[:nch]
    d_h = jnp.sum(d_sl * w_sl * N_sl)
    h_h = jnp.sum(w_sl * w_sl * N_sl)
    return d_h, h_h


# --------------------------------------------------------------------
# gb_wdm_get_ll  (per-binary)
# --------------------------------------------------------------------
def _get_ll_one_binary(params: jnp.ndarray, data_index: int, noise_index: int,
                       source: JaxAmpPhaseSource,
                       orbits: OrbitsWrapJAX,
                       tdi_config: TDIConfigWrapJAX,
                       wdm_lookup: WaveletLookupTableWrapJAX,
                       wdm: WDMDomainWrapJAX,
                       T: float, t_ref: float, tdi_type: int,
                       deriv_delta_t: float, num_diff: int
                       ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """One binary's ``(d|h), (h|h)``."""
    w = _w_mn_for_one_binary(params, source, orbits, tdi_config, wdm_lookup,
                             T, t_ref, deriv_delta_t, num_diff=num_diff)
    nch_data = wdm.num_channel
    Nf_act = wdm.Nf_active
    Nt_act = wdm.Nt_active

    d = wdm.wdm_data.reshape(wdm.num_data, nch_data, Nf_act, Nt_act)[data_index]

    if tdi_type == TDI_XYZ:
        invC = wdm.wdm_noise.reshape(wdm.num_noise, nch_data, nch_data,
                                     Nf_act, Nt_act)[noise_index]
        return _inner_product_xyz(d, w, invC)
    else:
        invN = wdm.wdm_noise.reshape(wdm.num_noise, nch_data, Nf_act,
                                     Nt_act)[noise_index]
        nch_use = 3 if tdi_type == TDI_AET else 2
        return _inner_product_diag(d, w, invN, nch_use)


def gb_wdm_get_ll_jax(params: jnp.ndarray,
                      data_index: jnp.ndarray, noise_index: jnp.ndarray,
                      source: JaxAmpPhaseSource,
                      orbits: OrbitsWrapJAX,
                      tdi_config: TDIConfigWrapJAX,
                      wdm_lookup: WaveletLookupTableWrapJAX,
                      wdm: WDMDomainWrapJAX,
                      T: float, t_ref: float, tdi_type: int,
                      deriv_delta_t: float = 500.0,
                      num_diff: int = DEFAULT_NUM_DIFF
                      ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JAX analog of the former C++ ``gb_wdm_get_ll_wrap``: per-binary ``<d|h>``, ``<h|h>``.

    The inner product has unit prefactor (see the module docstring); with XYZ it uses
    the full ``3 x 3`` inverse covariance, with AET / AE the diagonal one.

    Args:
        params: ``(num_bin, nparams)`` or flat ``(num_bin * nparams,)`` physical
            parameters in ``source``'s layout.
        data_index: ``(num_bin,)`` int data slab of each binary in ``wdm.wdm_data``.
        noise_index: ``(num_bin,)`` int noise slab in ``wdm.wdm_noise``.
        source: The amp / phase source (e.g.
            :class:`~gbgpu.jax.sources.ucb.JaxUCBSource`).
        orbits: JAX orbits.
        tdi_config: JAX TDI configuration.
        wdm_lookup: The wavelet lookup table and active-band geometry.
        wdm: The data / noise domain (``wdm_data`` / ``wdm_noise`` active-band slabs).
        T: Observation time [s]; unused.
        t_ref: Time origin of the WDM pixel grid [s].
        tdi_type: ``TDI_XYZ`` (1), ``TDI_AET`` (2) or ``TDI_AE`` (3).
        deriv_delta_t: Central-difference half-step [s] of the instantaneous-frequency
            estimate in ``fast_wdm_inner_jax``. Default ``500.0``.
        num_diff: Layers scattered each side of the anchor layer. Default ``2``.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: ``(d_h, h_h)``, each ``(num_bin,)``.
    """
    p = jnp.asarray(params)
    if p.ndim == 1:
        p = p.reshape(-1, source.n_params)
    num_bin = p.shape[0]

    def per_bin(pi, di, ni):
        return _get_ll_one_binary(pi, di, ni, source, orbits, tdi_config,
                                  wdm_lookup, wdm, T, t_ref, tdi_type,
                                  deriv_delta_t, num_diff)
    d_h_arr, h_h_arr = jax.vmap(per_bin)(p, data_index, noise_index)
    return d_h_arr, h_h_arr


# --------------------------------------------------------------------
# gb_wdm_fill_global  (per-binary; scatters into shared template buffer)
# --------------------------------------------------------------------
def gb_wdm_fill_global_jax(templates: jnp.ndarray,
                           params: jnp.ndarray,
                           data_index: jnp.ndarray,
                           factors: jnp.ndarray,
                           source: JaxAmpPhaseSource,
                           orbits: OrbitsWrapJAX,
                           tdi_config: TDIConfigWrapJAX,
                           wdm_lookup: WaveletLookupTableWrapJAX,
                           wdm: WDMDomainWrapJAX,
                           T: float, t_ref: float, tdi_type: int,
                           deriv_delta_t: float = 500.0,
                           num_diff: int = DEFAULT_NUM_DIFF
                           ) -> jnp.ndarray:
    """JAX analog of the former C++ ``gb_wdm_fill_global_wrap``.

    ``templates`` shape: ``(num_templates, nchannels, Nf_active,
    Nt_active)`` (or any shape that reshapes to it).

    Because JAX arrays are immutable, this *returns* the updated
    template buffer rather than mutating in place.

    Args:
        templates: The template buffer (active-band layout).
        params: ``(num_bin, nparams)`` or flat physical parameters.
        data_index: ``(num_bin,)`` int template slab each binary is added to.
        factors: ``(num_bin,)`` multipliers (``+1`` add, ``-1`` remove).
        source: The amp / phase source.
        orbits: JAX orbits.
        tdi_config: JAX TDI configuration.
        wdm_lookup: The wavelet lookup table and active-band geometry.
        wdm: The domain (its ``num_channel``, ``Nf_active``, ``Nt_active`` are read).
        T: Observation time [s]; unused.
        t_ref: Time origin of the WDM pixel grid [s].
        tdi_type: Unused here (accepted for the C++ signature).
        deriv_delta_t: As in :func:`gb_wdm_get_ll_jax`. Default ``500.0``.
        num_diff: As in :func:`gb_wdm_get_ll_jax`. Default ``2``.

    Returns:
        jnp.ndarray: ``templates + sum_i factors[i] * w_i`` (float64), in the input shape.
    """
    p = jnp.asarray(params)
    if p.ndim == 1:
        p = p.reshape(-1, source.n_params)
    num_bin = p.shape[0]

    nch = wdm.num_channel
    Nf_act = wdm.Nf_active
    Nt_act = wdm.Nt_active
    tmpl = templates.reshape(-1, nch, Nf_act, Nt_act).astype(jnp.float64)
    num_templates = tmpl.shape[0]

    def per_bin_w(pi):
        return _w_mn_for_one_binary(pi, source, orbits, tdi_config,
                                    wdm_lookup, T, t_ref, deriv_delta_t,
                                    num_diff=num_diff)
    w_all = jax.vmap(per_bin_w)(p)   # (num_bin, nch, Nf_act, Nt_act)

    contributions = w_all * factors[:, None, None, None]
    summed = jax.ops.segment_sum(contributions, data_index,
                                 num_segments=num_templates)
    out = tmpl + summed
    return out.reshape(templates.shape)


# --------------------------------------------------------------------
# gb_wdm_swap_ll  (per-binary; 5-way accumulators)
# --------------------------------------------------------------------
def _swap_ll_one(params_add: jnp.ndarray, params_rem: jnp.ndarray,
                 data_index: int, noise_index: int,
                 source: JaxAmpPhaseSource,
                 orbits: OrbitsWrapJAX,
                 tdi_config: TDIConfigWrapJAX,
                 wdm_lookup: WaveletLookupTableWrapJAX,
                 wdm: WDMDomainWrapJAX,
                 T: float, t_ref: float, tdi_type: int,
                 deriv_delta_t: float, num_diff: int
                 ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """One binary pair's ``(d_h_add, d_h_remove, add_add, remove_remove, add_remove)``."""
    # The former C++ swap kernel scattered into the *union* of the add and
    # remove m-windows. _w_mn_for_one_binary
    # already produces a full (nch, Nf_act, Nt_act) buffer for one
    # source with the num_diff window applied per pixel, so the union
    # is simply ``w_add + w_remove`` -- masking handled by the
    # ``in_band`` flag inside _w_mn_for_one_binary. We build the two
    # full buffers and combine via einsum on the dense inner-product
    # axes; this matches the C++ accumulators exactly because pixels
    # not in either window contribute zero from both buffers.
    w_a = _w_mn_for_one_binary(params_add, source, orbits, tdi_config,
                               wdm_lookup, T, t_ref, deriv_delta_t,
                               num_diff=num_diff)
    w_r = _w_mn_for_one_binary(params_rem, source, orbits, tdi_config,
                               wdm_lookup, T, t_ref, deriv_delta_t,
                               num_diff=num_diff)

    nch_data = wdm.num_channel
    Nf_act = wdm.Nf_active
    Nt_act = wdm.Nt_active
    d = wdm.wdm_data.reshape(wdm.num_data, nch_data, Nf_act, Nt_act)[data_index]

    if tdi_type == TDI_XYZ:
        invC = wdm.wdm_noise.reshape(wdm.num_noise, nch_data, nch_data,
                                     Nf_act, Nt_act)[noise_index]
        d_h_a, aa = _inner_product_xyz(d, w_a, invC)
        d_h_r, rr = _inner_product_xyz(d, w_r, invC)
        ar = jnp.einsum('i...,ij...,j...->', w_a, invC, w_r)
    else:
        invN = wdm.wdm_noise.reshape(wdm.num_noise, nch_data, Nf_act,
                                     Nt_act)[noise_index]
        nch_use = 3 if tdi_type == TDI_AET else 2
        d_h_a, aa = _inner_product_diag(d, w_a, invN, nch_use)
        d_h_r, rr = _inner_product_diag(d, w_r, invN, nch_use)
        ar = jnp.sum(w_a[:nch_use] * w_r[:nch_use] * invN[:nch_use])

    return d_h_a, d_h_r, aa, rr, ar


def gb_wdm_swap_ll_jax(params_add: jnp.ndarray, params_rem: jnp.ndarray,
                       data_index: jnp.ndarray, noise_index: jnp.ndarray,
                       source: JaxAmpPhaseSource,
                       orbits: OrbitsWrapJAX,
                       tdi_config: TDIConfigWrapJAX,
                       wdm_lookup: WaveletLookupTableWrapJAX,
                       wdm: WDMDomainWrapJAX,
                       T: float, t_ref: float, tdi_type: int,
                       deriv_delta_t: float = 500.0,
                       num_diff: int = DEFAULT_NUM_DIFF
                       ):
    """JAX analog of the former C++ ``gb_wdm_swap_ll_wrap``: the five swap inner products.

    Args:
        params_add: ``(num_bin, nparams)`` or flat parameters of the added sources.
        params_rem: Same for the removed sources.
        data_index: ``(num_bin,)`` int data slabs.
        noise_index: ``(num_bin,)`` int noise slabs.
        source: The amp / phase source.
        orbits: JAX orbits.
        tdi_config: JAX TDI configuration.
        wdm_lookup: The wavelet lookup table and active-band geometry.
        wdm: The data / noise domain.
        T: Observation time [s]; unused.
        t_ref: Time origin of the WDM pixel grid [s].
        tdi_type: ``TDI_XYZ`` / ``TDI_AET`` / ``TDI_AE``.
        deriv_delta_t: As in :func:`gb_wdm_get_ll_jax`. Default ``500.0``.
        num_diff: As in :func:`gb_wdm_get_ll_jax`. Default ``2``.

    Returns:
        tuple: ``(d_h_add, d_h_remove, add_add, remove_remove, add_remove)``, each of
        shape ``(num_bin,)``.
    """
    pa = jnp.asarray(params_add)
    pr = jnp.asarray(params_rem)
    if pa.ndim == 1:
        pa = pa.reshape(-1, source.n_params)
        pr = pr.reshape(-1, source.n_params)

    def per_bin(pai, pri, di, ni):
        return _swap_ll_one(pai, pri, di, ni, source, orbits, tdi_config,
                            wdm_lookup, wdm, T, t_ref, tdi_type,
                            deriv_delta_t, num_diff)
    return jax.vmap(per_bin)(pa, pr, data_index, noise_index)
