"""Shared fixture for the GB psd-mirror kernel tests + goldens recorder.

The grid is the SAME small grid as ``tests/test_sighet_inmodel_window.py``
(``_Grid._build_grid``: Nf=256, Nt=512, dt=10 s, two picked sources, W=5
narrow slabs with slot 1 deliberately offset so the candidate's active
m-band runs off the slab edge). It is re-stated here with a ``backend``
parameter so ``scripts/record_psd_mirror_goldens.py`` can record the CPU
goldens on the laptop AND the GPU goldens on the cluster from one source
of truth, and so the mirror tests can build a 2-walker parent psd plane
with DISTINCT invC per walker (walker 1 = 1.7 x walker 0 plus off-diagonal
terms) -- a wrong walker row or origin is then visible in every kernel.

Nothing here imports the sig-het comp; the chunked ``GBWDMComputations``
is the only engine, so the fixture is a few tens of MB, not the 2.7 GB
sig-het stash.
"""

import numpy as np

from lisatools.detector import ESAOrbits
from lisatools.domains import WDMSettings
from lisatools.utils.constants import YRSID_SI

from gbgpu.gbcomps import GBWDMComputations


class SlotHolder:
    """Minimal wdm_holder: N buffer slots (residual slab + XYZ invC slab).

    Mirror of ``test_sighet_inmodel_window._SlotHolder`` with an ``xp``.
    """

    def __init__(self, xp, data_slabs, invC_slabs):
        self.linear_data_arr = [xp.ascontiguousarray(xp.asarray(data_slabs)).ravel()]
        self.linear_psd_arr = [xp.ascontiguousarray(xp.asarray(invC_slabs)).ravel()]

    def __len__(self):
        return 1


class NarrowSlabHolder(SlotHolder):
    """Production layout: per-slot narrow band slabs (``band_slab_Nf`` +
    absolute per-slot origins ``slab_min_f``)."""

    def __init__(self, xp, data_slabs, invC_slabs, slab_Nf, slab_min_f):
        super().__init__(xp, data_slabs, invC_slabs)
        self.band_slab_Nf = int(slab_Nf)
        self.slab_min_f = xp.asarray(np.asarray(slab_min_f, dtype=np.int32))


class MirrorSlotHolder:
    """A holder in MIRROR layout: per-slot data slabs, but ``linear_psd_arr``
    is the parent's per-WALKER full-active-band psd plane and each slot
    carries its walker row in ``psd_row_index``.

    ``slab_Nf``/``slab_min_f`` may be ``None`` for a full-band data layout
    (each slot spans the whole active band, origin ``ind_min_f``).
    """

    def __init__(self, xp, data_slabs, mirror, psd_row_index,
                 slab_Nf=None, slab_min_f=None):
        self.linear_data_arr = [xp.ascontiguousarray(xp.asarray(data_slabs)).ravel()]
        self.linear_psd_arr = [xp.ascontiguousarray(xp.asarray(mirror)).ravel()]
        self.psd_row_index = xp.asarray(np.asarray(psd_row_index, dtype=np.int32))
        if slab_Nf is not None:
            self.band_slab_Nf = int(slab_Nf)
            self.slab_min_f = xp.asarray(np.asarray(slab_min_f, dtype=np.int32))

    def __len__(self):
        return 1


def per_walker_invC(nwalkers, nch, nfa, nta):
    """(nwalkers, nch, nch, nfa, nta) real symmetric invC planes, DISTINCT
    per walker so a wrong row is visible: walker w = (1 + 0.7 w) on the
    diagonal, with a small (m, n)-dependent off-diagonal term."""
    m = np.arange(nfa, dtype=float)[:, None]
    n = np.arange(nta, dtype=float)[None, :]
    ripple = 1.0 + 0.05 * np.cos(0.3 * m + 0.01 * n)
    out = np.zeros((nwalkers, nch, nch, nfa, nta))
    for w in range(nwalkers):
        scale = 1.0 + 0.7 * w
        for c in range(nch):
            out[w, c, c] = scale * ripple
        for c1 in range(nch):
            for c2 in range(c1 + 1, nch):
                od = 0.05 * scale * (c1 + 1) * np.sin(0.2 * m + 0.02 * n + w)
                out[w, c1, c2] = od
                out[w, c2, c1] = od
    return out


def build_fixture(backend="cpu", nwalkers=2):
    """Build the grid, comp, sources and every holder layout.

    Returns a dict with keys: ``xp, backend, wdm_set, chunked, layer_df,
    params_ref, params, params_remove, di, slots, psd_rows, slab_Nf,
    slab_min_f, Nf_active, Nt_active, ind_min_f, slabs_full (host, (2,3,
    Nf_active,Nt_active)), slabs_narrow (host, (2,3,W,Nt_active)), mirror
    (host, (nwalkers,3,3,Nf_active,Nt_active)), holders`` where ``holders``
    holds ``narrow_perslot``, ``narrow_mirror``, ``full_perslot``,
    ``full_mirror`` (per-slot invC picked from the mirror by (row, origin),
    so per-slot vs mirror must be bit-identical).
    """
    dt = 10.0
    Nf, Nt = 256, 512
    t_start = int(0.5 * YRSID_SI / dt) * dt
    layer_df = 1.0 / (2.0 * Nf * dt)
    edge = 40

    orbits = ESAOrbits(force_backend=backend)
    # Pre-configure the orbits on a SHORT, coarse grid covering the
    # observation window plus the chunk padding. Left unconfigured,
    # GBWDMComputations' orbits setter builds the dense
    # LINEAR_INTERP_TIMESTEP (50 s) grid over the WHOLE orbit file (5.6e6
    # points, ~3.0 GB maxrss -- the sig-het _Grid fixture's known
    # footprint), which the 8 GB laptop's 2.5 GB watchdog cannot host. The
    # orbit model is irrelevant to what these tests pin (invC ADDRESSING):
    # both layouts see the same orbits, and the goldens are recorded with
    # this same fixture (2 MB instead of 3 GB).
    T = Nf * Nt * dt
    o_dt = 1000.0
    margin = 64 * Nf * dt          # >> n_pad (16) layers of chunk padding
    t_arr = np.arange(-margin, T + margin + o_dt, o_dt) + t_start
    # The explicit ``t_arr`` branch of ``Orbits._configure`` (NOT
    # ``linear_interp_setup``, which ignores ``t_arr`` and builds the dense
    # full-file grid) splines the file onto this window and hands the C++
    # ``Orbits`` its start time / step / length. With the pre-2026-09-09
    # LAT that start time was the FILE's ``t0`` (0), not ``t_arr[0]`` (0.5
    # yr), so the C++ window lookup ``int((t - t0) / dt)`` fell past the
    # last node for every evaluation time -> zero unit vectors -> EXACTLY
    # ZERO templates: the first goldens were 72/74 all-zero arrays and every
    # mirror-parity test passed vacuously (0 == 0); the recorder's own
    # "0 nonzero outputs" line was the tell. LAT now passes ``t_arr[0]``;
    # ``build_fixture`` asserts non-zero templates below so a regression
    # can never again pass silently.
    orbits._configure(t_arr=t_arr)
    wdm_set = WDMSettings(
        Nf, Nt, dt, t0=t_start,
        min_freq=1e-4, max_freq=2e-2,
        min_time=edge * Nf * dt, max_time=(Nt - edge) * Nf * dt,
        force_backend=backend,
    )
    chunked = GBWDMComputations(
        wdm_set, t_ref=t_start,
        Nt_sub=128, n_pad=16, N_sparse=256,
        N_cp_sig=0, N_cp_orbit=0,
        orbits=orbits, tdi_config="2nd generation",
        force_backend=backend, d_d=0.0, tdi_type="XYZ",
    )
    chunked.convert_to_ra_dec = False
    xp = chunked.xp

    f0_A = (int(3e-3 / layer_df) + 0.37) * layer_df
    f0_C = (int(5e-3 / layer_df) + 0.62) * layer_df
    A = np.array([1e-21, f0_A, 1e-17, 0.0, 1.2, 0.7, 0.4, 2.0, 0.5])
    C = np.array([8e-22, f0_C, 2e-17, 0.0, 0.4, 1.1, 0.9, 4.0, -0.3])
    params_ref = np.stack([A, C])

    ilo, ihi = int(wdm_set.ind_min_f), int(wdm_set.ind_max_f) + 1
    slabs = []
    for p in (A, C):
        h = xp.zeros((3, Nf, Nt))
        chunked.fill_global_wdm(xp.asarray(p[None, :]), h, convert_to_ra_dec=False)
        h_host = np.asarray(h.get() if hasattr(h, "get") else h)
        h_act = np.ascontiguousarray(h_host[:, ilo:ihi, wdm_set.active_slice_t])
        slabs.append(h_act)
    slabs_full = np.stack(slabs)
    nch, nfa, nta = slabs[0].shape
    # Both reference templates must be non-trivial in EVERY channel: the
    # parity tests compare kernel outputs between two invC layouts, and
    # all-zero templates make every such comparison pass vacuously.
    per_ch = np.abs(slabs_full).reshape(2, nch, -1).max(axis=-1)
    if not np.all(per_ch > 0.0):
        raise RuntimeError(
            "psd_mirror_fixture: zero reference template(s) -- per-source, "
            f"per-channel max|h| = {per_ch.tolist()}; the orbit grid handed "
            "to the C++ response does not cover the observation window "
            "(see the Orbits._configure note above)")

    # per-walker parent psd plane (the thing the mirror replicates)
    mirror = per_walker_invC(nwalkers, nch, nfa, nta)
    # slot -> walker row; slot 0 on walker 1, slot 1 on walker 0 (NOT the
    # identity, so a "row == slot" bug is visible)
    psd_rows = np.array([1 % nwalkers, 0], dtype=np.int32)

    # ---- narrow-slab layout (W = 5, slot 1 offset off its carrier) --------
    slab_Nf = 5
    m_car = np.floor(params_ref[:, 1] / layer_df).astype(int)
    slab_min_f = np.array([m_car[0] - 2, m_car[1] - 4], dtype=np.int32)
    lo = slab_min_f - ilo
    assert (lo >= 0).all() and (lo + slab_Nf <= nfa).all()
    slabs_narrow = np.stack([slabs_full[i][:, lo[i]:lo[i] + slab_Nf, :]
                             for i in range(2)])
    invC_narrow = np.stack([
        mirror[psd_rows[i]][:, :, lo[i]:lo[i] + slab_Nf, :] for i in range(2)])
    invC_full = np.stack([mirror[psd_rows[i]] for i in range(2)])

    holders = dict(
        narrow_perslot=NarrowSlabHolder(xp, slabs_narrow, invC_narrow,
                                        slab_Nf, slab_min_f),
        narrow_mirror=MirrorSlotHolder(xp, slabs_narrow, mirror, psd_rows,
                                       slab_Nf, slab_min_f),
        full_perslot=SlotHolder(xp, slabs_full, invC_full),
        full_mirror=MirrorSlotHolder(xp, slabs_full, mirror, psd_rows),
    )

    # Scoring batch: references + jittered copies (same rng as the sig-het
    # window fixture so both suites score the same rows).
    rng = np.random.default_rng(20260909)
    rows = [A, C]
    for _ in range(2):
        for p in (A, C):
            q = p.copy()
            q[0] *= 1.0 + 0.15 * rng.standard_normal()
            q[1] += 0.05 * layer_df * rng.standard_normal()
            q[4] = rng.uniform(0.0, 2 * np.pi)
            rows.append(q)
    params = np.stack(rows)
    di = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)
    params_remove = params.copy()
    params_remove[:, 0] *= 0.9
    params_remove[:, 1] += 0.02 * layer_df
    params_remove[:, 4] += 0.7

    return dict(
        xp=xp, backend=backend, wdm_set=wdm_set, chunked=chunked,
        layer_df=layer_df, params_ref=params_ref, params=params,
        params_remove=params_remove, di=di,
        slots=np.array([0, 1], dtype=np.int32), psd_rows=psd_rows,
        slab_Nf=slab_Nf, slab_min_f=slab_min_f,
        Nf_active=nfa, Nt_active=nta, ind_min_f=ilo, nwalkers=nwalkers,
        slabs_full=slabs_full, slabs_narrow=slabs_narrow, mirror=mirror,
        holders=holders,
    )


def _host(a):
    return np.asarray(a.get() if hasattr(a, "get") else a)


def run_all_entry_points(fx, holder, m_half=1, fill=True):
    """Every chunked-het entry point on ``holder``; host-side copies.

    Returns a dict of arrays: ``ll, d_h, h_h, d_h_im`` (get_ll),
    ``sw_like_add, sw_like_rem, sw_dha, sw_dhr, sw_aa, sw_rr, sw_ar,
    sw_dha_im, sw_ar_im`` (swap), ``fs0_N, fs0_M, fs1_N, fs1_M`` (fstat
    fold 0/1) and, with ``fill``, ``fill`` (fill_global into a fresh buffer
    of the holder's data layout).
    """
    xp = fx["xp"]
    ch = fx["chunked"]
    params = xp.asarray(fx["params"])
    params_rm = xp.asarray(fx["params_remove"])
    di = xp.asarray(fx["di"])
    out = {}

    ll = ch.get_ll_wdm(params, holder, data_index=di, noise_index=di,
                       m_band_half_width=m_half)
    out["ll"] = _host(ll).copy()
    out["d_h"] = _host(ch.d_h_out).copy()
    out["h_h"] = _host(ch.h_h_out).copy()
    out["d_h_im"] = _host(ch.d_h_im_out).copy()

    sw = ch.get_swap_ll_wdm(params, params_rm, holder, data_index=di,
                            noise_index=di, m_band_half_width=m_half)
    for k, v in zip(("sw_like_add", "sw_like_rem", "sw_dha", "sw_dhr",
                     "sw_aa", "sw_rr", "sw_ar"), sw):
        out[k] = _host(v).copy()
    out["sw_dha_im"] = _host(ch.d_h_add_im_out).copy()
    out["sw_ar_im"] = _host(ch.add_remove_im_out).copy()

    for fold in (0, 1):
        N, M = ch.get_fstat_ll_wdm(params, holder, data_index=di,
                                   noise_index=di, m_band_half_width=m_half,
                                   fstat_fold=fold)
        out[f"fs{fold}_N"] = _host(N).copy()
        out[f"fs{fold}_M"] = _host(M).copy()

    if fill:
        n_slots = 2
        if getattr(holder, "band_slab_Nf", None) is not None:
            W = int(holder.band_slab_Nf)
            buf = xp.zeros(n_slots * 3 * W * fx["Nt_active"])
            ch.fill_global_wdm(params, buf, data_index=di,
                               m_band_half_width=m_half,
                               band_slab_Nf=W, slab_min_f=holder.slab_min_f)
        else:
            buf = xp.zeros(n_slots * 3 * fx["Nf_active"] * fx["Nt_active"])
            ch.fill_global_wdm(params, buf, data_index=di,
                               m_band_half_width=m_half)
        out["fill"] = _host(buf).copy()
    return out
